"""classifier_model_router: one capsule, two providers (claude, chatgpt), selected by an option."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "capsules"))

from vcap import DetectionNode, rect_to_coords  # noqa: E402

from classifier_model_router import backend as backend_module  # noqa: E402
from classifier_model_router.backend import Backend, build_request, model_of, provider_of, reply_text  # noqa: E402
from classifier_model_router.capsule import Capsule  # noqa: E402
from classifier_model_router.stream_state import StreamState  # noqa: E402


def options(**changes):
    values = {name: option.default for name, option in Capsule.options.items()}
    values.update(claude_api_key="sk-ant-test", chatgpt_api_key="sk-oai-test", detection_interval=60, current_count=10)
    values.update(changes)
    return values


def person(x1, y1, x2, y2):
    return DetectionNode(name="person", coords=rect_to_coords([x1, y1, x2, y2]))


class Reply:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {}, text

    def json(self):
        return self._body


def claude_answer(text):
    return {"content": [{"type": "thinking", "thinking": "", "signature": "x"}, {"type": "text", "text": text}], "stop_reason": "end_turn"}


def chatgpt_answer(text):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


class Now:
    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def shutdown(self, wait=True):
        pass


def run(reply, people=None, state=None, **option_changes):
    state, backend = state or StreamState(), Backend()
    backend.executor = Now()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    people = [person(100, 100, 300, 500)] if people is None else people
    with mock.patch.object(backend_module.requests, "post", return_value=reply) as post:
        result = backend.process_frame(frame, people, options(**option_changes), state)
    return result, post.call_args_list, state


def test_only_the_keys_and_the_proxy_are_empty_by_default_and_each_provider_has_its_cheapest_current_model():
    defaults = {name: option.default for name, option in Capsule.options.items()}
    assert [name for name, value in defaults.items() if value in ("", None)] == ["claude_api_key", "chatgpt_api_key", "http_proxy"]
    assert defaults["provider"] == "claude" and defaults["claude_model"] == "claude-haiku-5-5" and defaults["chatgpt_model"] == "gpt-5-nano"
    assert model_of(options()) == "claude-haiku-5-5"
    assert model_of(options(provider="chatgpt")) == "gpt-5-nano"
    assert model_of(options(provider="chatgpt", chatgpt_model="gpt-5.4-nano")) == "gpt-5.4-nano"
    assert model_of(options(claude_model="claude-sonnet-5-5")) == "claude-sonnet-5-5"
    assert model_of(options(claude_model=" ")) == "claude-haiku-5-5"  # emptied by hand: the cheapest model again
    assert provider_of(options(provider=" OpenAI ")) == "chatgpt" and provider_of(options(provider="Anthropic")) == "claude"
    with pytest.raises(ValueError, match="unknown provider 'gemini'"):
        provider_of(options(provider="gemini"))


def test_the_claude_request():
    url, headers, body = build_request("AAAA", options())
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "sk-ant-test" and headers["anthropic-version"] == "2023-06-01" and "Authorization" not in headers
    assert body["model"] == "claude-haiku-5-5" and body["output_config"] == {"effort": "low"} and body["max_tokens"] == 1024
    assert not {"temperature", "top_p", "top_k"} & set(body)
    image, text = body["messages"][0]["content"]
    assert image["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"} and text["text"].startswith("How many people in the picture are holding a phone?")
    assert "output_config" not in build_request("AAAA", options(effort=""))[2]


def test_the_chatgpt_request_for_a_gpt5_model_and_for_an_older_one():
    url, headers, body = build_request("AAAA", options(provider="chatgpt"))
    assert url == "https://api.openai.com/v1/chat/completions" and headers["Authorization"] == "Bearer sk-oai-test"
    assert body["model"] == "gpt-5-nano" and body["max_completion_tokens"] == 1024 and body["reasoning_effort"] == "low"
    assert not {"temperature", "top_p", "max_tokens"} & set(body)
    assert body["messages"][0]["content"][1]["image_url"] == {"url": "data:image/jpeg;base64,AAAA", "detail": "low"}
    old = build_request("AAAA", options(provider="chatgpt", chatgpt_model="gpt-4o", effort=""))[2]
    assert old["max_tokens"] == 1024 and not {"max_completion_tokens", "reasoning_effort", "temperature"} & set(old)


def test_each_reply_format_is_read_by_its_own_provider():
    assert reply_text("claude", claude_answer("True")) == "True"  # after a thinking block
    assert reply_text("chatgpt", chatgpt_answer(" 2 ")) == "2"
    assert reply_text("claude", chatgpt_answer("x")) == "" and reply_text("chatgpt", claude_answer("x")) == ""


@pytest.mark.parametrize("provider,answer,url", [
    ("claude", claude_answer("True"), "https://api.anthropic.com/v1/messages"),
    ("chatgpt", chatgpt_answer("True"), "https://api.openai.com/v1/chat/completions"),
])
def test_the_selected_provider_is_called_and_the_answer_is_on_the_box_of_the_person(provider, answer, url):
    people = [person(100, 100, 300, 500)]
    result, calls, state = run(Reply(body=answer), people=people, provider=provider)
    state.annotate(people, 60)  # the next frame
    (call,) = calls
    assert call.args[0] == url and call.kwargs["timeout"] == 60
    assert result == []  # the capsule has no detections of its own: the answer goes into the person
    assert people[0].class_name == "person" and people[0].extra_data == {"model_router": "True"}


def test_an_unknown_provider_or_a_missing_key_of_the_selected_provider_makes_no_call():
    for changes in ({"provider": "gemini"}, {"claude_api_key": ""}, {"provider": "chatgpt", "chatgpt_api_key": ""}):
        result, calls, state = run(Reply(body=chatgpt_answer("True")), **changes)
        assert result == [] and calls == [] and state.get_last_detection_timestamp() > 0


@pytest.mark.parametrize("provider", ["claude", "chatgpt"])
@pytest.mark.parametrize("reply", [
    Reply(401, text="bad key"),
    Reply(200, body={"content": [], "choices": []}),
])
def test_a_failed_call_is_not_repeated_for_every_frame(provider, reply):
    result, calls, state = run(reply, detection_interval=0, provider=provider)
    assert result == [] and len(calls) == 1 and state.get_pending() == 0
    again, calls, _ = run(reply, state=state, detection_interval=0, provider=provider)
    assert again == [] and calls == []


def test_the_biggest_people_up_to_the_limit_are_sent_and_the_quota_counts_calls():
    people = [person(0, 0, 100, 100), person(200, 0, 400, 200), person(500, 0, 800, 300)]
    opts = options(current_count=3, detection_interval=0, max_detections_per_frame=2)
    state, backend = StreamState(), Backend()
    backend.executor = Now()
    sent = []
    with mock.patch.object(backend_module.requests, "post", side_effect=lambda *a, **k: sent.append(k["json"]) or Reply(body=claude_answer("1"))):
        results = [backend.process_frame(np.zeros((1080, 1920, 3), np.uint8), people, opts, state) for _ in range(3)]
    assert len(sent) == 3 and opts["current_count"] == 0  # 3 calls, not 3 frames
    assert state.annotate(people, 60) == 2  # the two biggest people got their answers
    assert [p.extra_data for p in people] == [{}, {"model_router": "1"}, {"model_router": "1"}]


def test_a_large_crop_is_sent_smaller():
    import base64
    import cv2

    _, calls, _ = run(Reply(body=claude_answer("0")), people=[person(0, 0, 1900, 1000)])
    data = calls[0].kwargs["json"]["messages"][0]["content"][0]["source"]["data"]
    sent = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    assert max(sent.shape[:2]) == 512


def test_nothing_new_is_sent_while_an_answer_is_missing_and_the_interval_is_kept():
    state = StreamState()
    state.request_started()
    assert run(Reply(body=claude_answer("x")), state=state, detection_interval=0)[1] == []
    _, first, state2 = run(Reply(body=claude_answer("x")))
    assert len(first) == 1 and run(Reply(body=claude_answer("x")), state=state2)[1] == []


def test_an_answer_stays_on_its_person_who_may_move_until_it_is_old():
    people = [person(100, 100, 300, 500), person(900, 100, 1100, 500)]
    _, _, state = run(Reply(body=claude_answer("1")), people=people[:1])
    moved = [person(120, 110, 320, 510), person(900, 100, 1100, 500)]  # the first person moved a little
    assert state.annotate(moved, 60) == 1
    assert moved[0].extra_data == {"model_router": "1"} and moved[1].extra_data == {}
    far = [person(1500, 100, 1700, 500)]  # nobody is there any more: the answer is not given to someone else
    assert state.annotate(far, 60) == 0 and far[0].extra_data == {}
    state.answers[0]["at"] -= 10000
    assert state.annotate(moved, 60) == 0  # too old
