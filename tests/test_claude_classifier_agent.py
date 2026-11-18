"""claude_classifier_agent: the request it sends, how it reads the reply, and that a failure does not cost a call per frame."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "capsules"))

from claude_classifier_agent import backend as backend_module  # noqa: E402
from claude_classifier_agent.backend import Backend  # noqa: E402
from claude_classifier_agent.capsule import Capsule  # noqa: E402
from claude_classifier_agent.stream_state import StreamState  # noqa: E402


def options(**changes):
    values = {name: option.default for name, option in Capsule.options.items()}
    values.update(api_key="sk-ant-test", detection_interval=60)
    values.update(changes)
    return values


class Reply:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


def answer(text, **extra):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", **extra}


@pytest.fixture
def run():
    """process_frame with the network replaced by a canned reply; returns (detections, requests sent, state)."""
    def go(reply, frame=None, state=None, **option_changes):
        state = state or StreamState()
        frame = np.zeros((1080, 1920, 3), np.uint8) if frame is None else frame
        with mock.patch.object(backend_module.requests, "post", return_value=reply) as post:
            detections = Backend().process_frame(frame, None, options(**option_changes), state)
        return detections, post.call_args_list, state
    return go


def test_the_default_model_is_the_cheapest_current_one_and_there_is_no_temperature():
    assert Capsule.options["model"].default == "claude-haiku-5-5"
    assert "temperature" not in Capsule.options  # the current models only accept their default
    assert Capsule.options["effort"].default == "low" and Capsule.options["max_tokens"].default >= 1024


def test_the_request_has_the_current_shape(run):
    _, calls, _ = run(Reply(body=answer("True")))
    (call,) = calls
    body = call.kwargs["json"]
    assert call.args[0] == "https://api.anthropic.com/v1/messages" and call.kwargs["timeout"] == 60
    assert call.kwargs["headers"]["x-api-key"] == "sk-ant-test" and call.kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-haiku-5-5" and body["output_config"] == {"effort": "low"} and body["max_tokens"] == 1024
    assert not {"temperature", "top_p", "top_k"} & set(body)
    image, text = body["messages"][0]["content"]
    assert image["source"]["media_type"] == "image/jpeg" and text["text"].startswith("Does this person")


def test_a_model_without_effort_gets_no_effort(run):
    _, calls, _ = run(Reply(body=answer("True")), model="claude-haiku-4-5", effort="")
    assert "output_config" not in calls[0].kwargs["json"]


def test_the_text_of_the_reply_is_the_result_even_after_thinking_blocks(run):
    reply = answer("True")
    reply["content"].insert(0, {"type": "thinking", "thinking": "", "signature": "x"})
    detections, _, state = run(Reply(body=reply))
    assert len(detections) == 1 and detections[0].extra_data == {"claude": "True"}
    assert detections[0].coords == [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]  # the whole frame, in frame pixels
    assert state.get_last_detection_timestamp() > 0


def test_a_large_frame_is_sent_smaller_but_the_result_stays_in_frame_pixels(run):
    import base64
    import cv2

    detections, calls, _ = run(Reply(body=answer("0")), frame=np.zeros((2160, 3840, 3), np.uint8))
    data = calls[0].kwargs["json"]["messages"][0]["content"][0]["source"]["data"]
    sent = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    assert max(sent.shape[:2]) == 1568
    assert detections[0].coords[2] == [3840, 2160]


@pytest.mark.parametrize("reply", [
    Reply(401, text='{"type":"error","error":{"type":"authentication_error"}}'),
    Reply(200, body={"content": [{"type": "thinking", "thinking": ""}], "stop_reason": "max_tokens"}),
    Reply(200, body={"choices": [{"message": {"content": "an OpenAI shaped reply"}}]}),
])
def test_a_failed_call_gives_nothing_and_is_not_repeated_for_every_frame(run, reply):
    detections, _, state = run(reply, detection_interval=0)
    assert detections == []
    # even with an interval of 0 the capsule waits a few seconds before it tries again
    again, calls, _ = run(reply, state=state, detection_interval=0)
    assert again == [] and calls == []


def test_an_error_while_calling_is_logged_not_raised_and_not_repeated(run):
    state = StreamState()
    with mock.patch.object(backend_module.requests, "post", side_effect=OSError("network down")):
        assert Backend().process_frame(np.zeros((10, 10, 3), np.uint8), None, options(), state) == []
    assert state.get_last_detection_timestamp() > 0


def test_no_api_key_means_no_call(run):
    detections, calls, _ = run(Reply(body=answer("True")), api_key="")
    assert detections == [] and calls == []


def test_the_detection_interval_limits_the_calls(run):
    _, first, state = run(Reply(body=answer("True")))
    again, calls, _ = run(Reply(body=answer("True")), state=state)
    assert len(first) == 1 and again == [] and calls == []
