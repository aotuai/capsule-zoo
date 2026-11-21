"""Cut the class head of a model with thousands of classes (YOLOE prompt-free) down to the selected classes.

The prompt-free model scores every box against all of its 4,585 classes, which is about half of its inference time and 155 MB of
output per frame. When the ``classes`` option selects a few classes, the weights and the bias of the class head are cut to those
classes, so the model only computes and gives those: about twice as fast in the measurements of the tutorial.

The cut model is built and compiled in a background thread the first time a set of classes is selected; until it is ready the
frames run on the model that is active (the full one at first), with the classes selected in ``decode()``, so the results are the
same and only the speed changes. A model that cannot be cut (the head is not found, the compile fails) is logged and the capsule
goes on with the full model.
"""
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

#: with more selected classes than this the cut model is hardly faster than the full one
MAX_CUT_CLASSES = 1000
#: how many cut models are kept (the full model is always kept)
MAX_CACHED = 3

logger = logging.getLogger(__name__)


def cut_model(model, num_classes: int, selected: np.ndarray):
    """A copy of ``model`` whose class outputs are only the ``selected`` classes, in that order.

    The constants that have a dimension of ``num_classes`` (the weights and the bias of the class head) are cut along it, and the
    shape patterns of the reshapes that name ``num_classes`` are changed to the new number. The output keeps the box values before and the mask values after the classes.
    """
    import openvino.runtime.opset13 as ops

    selected = np.asarray(selected, dtype=np.int64)
    before = [list(o.get_partial_shape().get_min_shape()) for o in model.outputs]
    cut = model.clone()
    changed = 0
    for op in list(cut.get_ordered_ops()):
        if op.get_type_name() != "Constant":
            continue
        shape = list(op.get_output_shape(0))
        if op.get_element_type().is_real():  # a weight or a bias with a dimension of num_classes
            if num_classes not in shape:
                continue
            new = ops.constant(np.take(op.data, selected, axis=shape.index(num_classes)))
        else:  # a shape pattern that names the number of classes, for example the target of a Reshape: [1, 4585, 400]
            data = op.data
            if data.ndim != 1 or not (data == num_classes).any():
                continue
            patched = data.copy()
            patched[patched == num_classes] = len(selected)
            new = ops.constant(patched)
        for target in list(op.output(0).get_target_inputs()):
            target.replace_source_output(new.output(0))
        changed += 1
    if changed == 0:
        raise ValueError(f"no class head with {num_classes} classes found in the model")
    cut.validate_nodes_and_infer_types()
    for old, new in zip(before, [list(o.get_partial_shape().get_min_shape()) for o in cut.outputs]):
        if len(old) == 3 and new[1] != old[1] - num_classes + len(selected):
            raise ValueError(f"the cut model gives {new[1]} values per box, expected {old[1] - num_classes + len(selected)}")
    return cut


class ClassHeadCutter:
    """Keeps the cut models of one backend, builds them in the background and installs the one that fits the selection."""

    def __init__(self, backend, num_classes: int):
        self.backend = backend
        self.num_classes = num_classes
        self.full = (backend.compiled_model, backend.infer_queue, backend._total_requests)
        self.active_key: Optional[Tuple[int, ...]] = None  # None: the full model
        self.active_map: Optional[np.ndarray] = None  # class number of each class output of the active model, None for the full one
        self.ready: Dict[Tuple[int, ...], tuple] = {}  # key -> (compiled model, queue, requests), most recent last
        self.failed = set()
        self._building: Optional[Tuple[int, ...]] = None
        self._wanted: Optional[Tuple[int, ...]] = None
        self._lock = threading.Lock()

    def use(self, selected: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Install the cut model for ``selected`` when it is ready and return the class numbers of the active model's classes.

        ``None`` means the full model is active: its output has all the classes, and ``decode()`` takes ``selected``. Otherwise the
        active model gives only the classes of the returned array, in that order. Call this and run the frame in one critical
        section: the model must not change in between.
        """
        key = None
        if selected is not None and 0 < selected.size <= MAX_CUT_CLASSES:
            key = tuple(int(c) for c in np.unique(selected))  # the same classes in another order are the same model
        if key == self.active_key:
            return self.active_map
        if key is None:
            self._install(None, None)
            return None
        with self._lock:
            entry = self.ready.pop(key, None)
            if entry is not None:
                self.ready[key] = entry  # most recent last
            elif key not in self.failed:
                self._wanted = key
                if self._building is None:
                    self._start(key)
        if entry is not None:
            self._install(key, entry)
        return self.active_map

    def _install(self, key, entry) -> None:
        backend = self.backend
        compiled, queue, requests = self.full if key is None else entry
        with backend._get_free_request_lock:  # no frame is waiting for a result on the old queue
            backend.compiled_model, backend.infer_queue, backend._total_requests = compiled, queue, requests
        self.active_key = key
        self.active_map = None if key is None else np.array(key, dtype=np.int64)
        if key is not None:
            logger.info("The classes option selects %d classes: running the model with the class head cut to them", len(key))

    def _start(self, key) -> None:
        self._building = key
        # Not a daemon thread: the process waits for a compile that is running when it ends (a few seconds), instead of
        # stopping OpenVINO in the middle of it, which aborts the process
        threading.Thread(target=self._build, args=(key,), name="class-head-cut").start()

    def _build(self, key) -> None:
        started = time.time()
        try:
            from openvino.runtime import AsyncInferQueue

            backend = self.backend
            cut = cut_model(backend.model, self.num_classes, np.array(key, dtype=np.int64))
            compiled = backend.ie.compile_model(model=cut, device_name="CPU")
            try:
                requests = compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
            except RuntimeError:
                requests = 1
            queue = AsyncInferQueue(compiled, requests)
            queue.set_callback(backend.on_result)
            with self._lock:
                self.ready[key] = (compiled, queue, requests)
                while len(self.ready) > MAX_CACHED:
                    self.ready.pop(next(iter(self.ready)))
            logger.info("Built the model with %d classes in %.1f s", len(key), time.time() - started)
        except Exception as error:  # the capsule goes on with the full model
            logger.warning("The class head could not be cut to %d classes (%s: %s); using the full model",
                           len(key), type(error).__name__, error)
            with self._lock:
                self.failed.add(key)
        finally:
            with self._lock:
                self._building = None
                wanted, self._wanted = self._wanted, None
                if wanted is not None and wanted != key and wanted not in self.ready and wanted not in self.failed:
                    self._start(wanted)
