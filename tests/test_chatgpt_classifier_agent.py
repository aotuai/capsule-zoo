"""chatgpt_classifier_agent: the request for each kind of model, how the people are chosen, and the limits on calls."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "capsules"))

from vcap import DetectionNode, rect_to_coords  # noqa: E402

from chatgpt_classifier_agent import backend as backend_module  # noqa: E402
from chatgpt_classifier_agent.backend import Backend, build_request, person_boxes  # noqa: E402
from chatgpt_classifier_agent.capsule import Capsule  # noqa: E402
from chatgpt_classifier_agent.stream_state import StreamState  # noqa: E402


def options(**changes):
    values = {name: option.default for name, option in Capsule.options.items()}
    values.update(api_key="sk-test", detection_interval=60, current_count=10)
    values.update(changes)
    return values


def person(x1, y1, x2, y2):
    return DetectionNode(name="person", coords=rect_to_coords([x1, y1, x2, y2]))


class Reply:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


def answer(text):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


class Now:
    """A thread pool that runs the work at once, so the tests do not wait for threads."""
    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def shutdown(self, wait=True):
        pass


@pytest.fixture
def run():
    def go(reply, people=None, state=None, backend=None, **option_changes):
        state = state or StreamState()
        backend = backend or Backend()
        backend.executor = Now()
        frame = np.zeros((1080, 1920, 3), np.uint8)
        people = [person(100, 100, 300, 500)] if people is None else people
        with mock.patch.object(backend_module.requests, "post", return_value=reply) as post:
            result = backend.process_frame(frame, people, options(**option_changes), state)
        return result, post.call_args_list, state, backend
    return go


def test_the_default_model_is_the_cheapest_current_one():
    assert Capsule.options["model"].default == "gpt-5-nano"
    assert Capsule.options["reasoning_effort"].default == "minimal" and Capsule.options["image_detail"].default == "low"
    assert not {"input_detections", "output_attributes"} & set(Capsule.options)  # they were never used


def test_a_gpt5_model_gets_the_parameters_it_accepts():
    body = build_request("AAAA", options())
    assert body["model"] == "gpt-5-nano" and body["max_completion_tokens"] == 1024 and body["reasoning_effort"] == "minimal"
    assert not {"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"} & set(body)
    image = body["messages"][0]["content"][1]["image_url"]
    assert image["detail"] == "low" and image["url"] == "data:image/jpeg;base64,AAAA"
    assert "reasoning_effort" not in build_request("AAAA", options(reasoning_effort=""))


def test_an_older_model_keeps_the_old_parameters():
    body = build_request("AAAA", options(model="gpt-4o"))
    assert body["max_tokens"] == 1024 and body["temperature"] == 1.0
    assert not {"max_completion_tokens", "reasoning_effort"} & set(body)


def test_the_request_goes_to_the_api_with_a_timeout(run):
    result, calls, _, _ = run(Reply(body=answer("True")))
    (call,) = calls
    assert call.args[0] == "https://api.openai.com/v1/chat/completions" and call.kwargs["timeout"] == 60
    assert call.kwargs["headers"]["Authorization"] == "Bearer sk-test" and call.kwargs["proxies"] is None
    assert len(result) == 1 and result[0].extra_data == {"chatgpt": "True"}
    assert result[0].coords == rect_to_coords([100, 100, 300, 500])  # the box of the person, in frame pixels


def test_two_people_with_the_same_box_size_are_both_sent():
    frame_shape = (1080, 1920, 3)
    boxes = person_boxes([person(0, 0, 100, 100), person(500, 500, 600, 600), person(0, 0, 300, 300), person(0, 0, 5, 5)], 900, frame_shape)
    assert boxes == [[0, 0, 300, 300], [0, 0, 100, 100], [500, 500, 600, 600]]  # biggest first, the tiny one left out


def test_boxes_may_be_dicts_and_are_clipped_to_the_frame():
    dicts = [{"coords": rect_to_coords([-50, -20, 150, 100])}, {"coords": rect_to_coords([1800, 1000, 2400, 1500])}]
    assert person_boxes(dicts, 900, (1080, 1920, 3)) == [[0, 0, 150, 100], [1800, 1000, 1920, 1080]]  # sorted by the clipped area


def test_only_the_biggest_people_up_to_the_limit_are_sent(run):
    people = [person(0, 0, 100, 100), person(200, 0, 400, 200), person(500, 0, 800, 300)]
    result, calls, _, _ = run(Reply(body=answer("1")), people=people, max_detections_per_frame=2)
    assert len(calls) == 2 and [d.coords for d in result] == [rect_to_coords([500, 0, 800, 300]), rect_to_coords([200, 0, 400, 200])]


def test_a_large_crop_is_sent_smaller(run):
    import base64
    import cv2

    _, calls, _, _ = run(Reply(body=answer("0")), people=[person(0, 0, 1900, 1000)])
    url = calls[0].kwargs["json"]["messages"][0]["content"][1]["image_url"]["url"]
    sent = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert max(sent.shape[:2]) == 512  # detail "low": the API looks at 512 x 512 anyway


@pytest.mark.parametrize("reply", [
    Reply(401, text='{"error":{"code":"invalid_api_key"}}'),
    Reply(200, body={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}),
    Reply(200, body={"content": [{"type": "text", "text": "an Anthropic shaped reply"}]}),
])
def test_a_failed_call_gives_nothing_and_is_not_repeated_for_every_frame(run, reply):
    result, calls, state, backend = run(reply, detection_interval=0)
    assert result == [] and len(calls) == 1 and state.get_pending() == 0
    again, calls, _, _ = run(reply, state=state, backend=backend, detection_interval=0)
    assert again == [] and calls == []  # even with an interval of 0 the capsule waits a few seconds


def test_an_error_while_calling_is_logged_not_raised_and_the_slot_is_given_back():
    state, backend = StreamState(), Backend()
    backend.executor = Now()
    with mock.patch.object(backend_module.requests, "post", side_effect=OSError("network down")):
        backend.process_frame(np.zeros((1080, 1920, 3), np.uint8), [person(0, 0, 200, 200)], options(), state)
    assert state.get_pending() == 0 and state.get_last_detection_timestamp() > 0


def test_nothing_new_is_sent_while_an_answer_is_still_missing(run):
    state = StreamState()
    state.request_started()  # a request that has not come back
    result, calls, _, _ = run(Reply(body=answer("True")), state=state, detection_interval=0)
    assert calls == []


def test_no_api_key_means_no_call(run):
    _, calls, _, _ = run(Reply(body=answer("True")), api_key="")
    assert calls == []


def test_the_quota_counts_one_call_per_person_and_stops_at_zero(run):
    opts = options(current_count=3, detection_interval=0, max_detections_per_frame=2)
    state, backend = StreamState(), Backend()
    backend.executor = Now()
    people = [person(0, 0, 100, 100), person(200, 0, 400, 200)]
    sent = []
    with mock.patch.object(backend_module.requests, "post", side_effect=lambda *a, **k: sent.append(1) or Reply(body=answer("1"))):
        for _ in range(3):
            backend.process_frame(np.zeros((1080, 1920, 3), np.uint8), people, opts, state)
    assert len(sent) == 3 and opts["current_count"] == 0  # 3 calls, not 3 frames


def test_the_detection_interval_limits_the_calls(run):
    _, first, state, backend = run(Reply(body=answer("True")))
    _, calls, _, _ = run(Reply(body=answer("True")), state=state, backend=backend)
    assert len(first) == 1 and calls == []


def test_the_answers_are_returned_once():
    state = StreamState()
    state.set_detection_response("a")
    assert state.get_detection_response() == ["a"] and state.get_detection_response() == []
