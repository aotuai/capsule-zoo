"""The ``classes`` option of a capsule made from an open-vocabulary model (YOLO-World, YOLOE).

The model was packaged with a vocabulary of class names (``labels``). The option picks the names that the capsule
reports, so they can change without packaging again. Names are matched ignoring case and spaces at the ends.
"""
import logging
from typing import Dict, List, Optional

import numpy as np

_cache: Dict[str, Optional[np.ndarray]] = {}


def selected_classes(text: Optional[str], labels: List[str]) -> Optional[np.ndarray]:
    """The numbers (in ``labels``) of the names in ``text``, separated by commas; ``None`` means every class.

    A name that is not in the vocabulary is logged and skipped. A list with only unknown names selects nothing.
    """
    if text is None or not text.strip():
        return None
    if text not in _cache:
        if len(_cache) > 64:
            _cache.clear()
        index: Dict[str, int] = {}
        for number, name in enumerate(labels):
            index.setdefault(name.strip().lower(), number)
        chosen: List[int] = []
        for name in text.split(","):
            name = name.strip().lower()
            if not name:
                continue
            if name not in index:
                logging.getLogger(__name__).warning("The classes option has a name that is not in the model: %r", name)
            elif index[name] not in chosen:
                chosen.append(index[name])
        _cache[text] = np.array(chosen, dtype=np.int64)
        # one line each time the option text changes, so the log shows what the capsule really received
        logging.getLogger(__name__).info("The classes option is %r: %d of %d classes selected", text, len(chosen), len(labels))
    return _cache[text]
