import io
import json
import unittest

from rec_mlx.anthropic import MessageEventStream, chat_to_message, messages_to_chat


def chat(**overrides):
    body = {
        "model": "flint",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    body.update(overrides)
    converted, ignored, thinking = messages_to_chat(body)
    return converted, ignored, thinking


class MessagesRequestTests(unittest.TestCase):
    def test_maps_system_sampling_and_stop_sequences(self):
        converted, ignored, thinking = chat(
            system=[{"type": "text", "text": "Be brief", "cache_control": {"type": "ephemeral"}}],
            temperature=0.2,
            top_p=0.9,
            top_k=8,
            stop_sequences=["END"],
            metadata={"user_id": "a"},
            thinking={"type": "enabled", "budget_tokens": 100},
        )
        self.assertEqual(ignored, set())
        self.assertTrue(thinking)
        self.assertEqual(converted["messages"][0], {"role": "system", "content": "Be brief"})
        self.assertEqual(converted["temperature"], 0.2)
        self.assertEqual(converted["top_p"], 0.9)
        self.assertEqual(converted["top_k"], 8)
        self.assertEqual(converted["stop"], ["END"])
        self.assertNotIn("stream", converted)

    def test_maps_tools_tool_use_and_tool_results(self):
        converted, ignored, _ = chat(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Look it up", "signature": "sig"},
                        {"type": "text", "text": "Checking"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "sunny"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-2",
                            "content": [{"type": "text", "text": "12"}],
                            "is_error": True,
                        },
                        {"type": "text", "text": "Thanks"},
                    ],
                },
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            tool_choice={"type": "auto"},
        )
        self.assertEqual(ignored, set())
        self.assertEqual(
            converted["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        self.assertEqual(converted["tool_choice"], "auto")
        assistant, result, failed, thanks = converted["messages"]
        self.assertEqual(assistant["reasoning_content"], "Look it up")
        self.assertEqual(assistant["content"], "Checking")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call-1")
        self.assertEqual(
            json.loads(assistant["tool_calls"][0]["function"]["arguments"]), {"city": "Paris"}
        )
        self.assertEqual(result, {"role": "tool", "tool_call_id": "call-1", "content": "sunny"})
        self.assertEqual(failed["content"], "Error: 12")
        self.assertEqual(thanks, {"role": "user", "content": "Thanks"})

    def test_streaming_requests_usage_and_can_hide_thinking(self):
        converted, _, thinking = chat(stream=True, thinking={"type": "disabled"})
        self.assertFalse(thinking)
        self.assertEqual(converted["stream_options"], {"include_usage": True})

    def test_unknown_fields_are_reported_and_structural_errors_fail(self):
        _, ignored, _ = chat(service_tier="auto")
        self.assertEqual(ignored, {"service_tier"})
        for body, text in (
            ({"model": "flint", "messages": [{"role": "user", "content": "Hi"}]}, "max_tokens"),
            ({"model": "flint", "max_tokens": 4, "messages": []}, "nonempty"),
            (
                {
                    "model": "flint",
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "tool_choice": {"type": "tool", "name": "get_weather"},
                },
                "forced calls",
            ),
            (
                {
                    "model": "flint",
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}],
                },
                "image",
            ),
            (
                {
                    "model": "flint",
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "disable_parallel_tool_calls": True,
                },
                "not supported",
            ),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                messages_to_chat(body)


class MessagesResponseTests(unittest.TestCase):
    def test_maps_text_thinking_tool_use_and_stop_reasons(self):
        message = chat_to_message(
            {
                "id": "chatcmpl-abc",
                "model": "flint",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "Done",
                            "reasoning": "Because",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            }
        )
        self.assertEqual(message["id"], "msg_abc")
        self.assertEqual(message["stop_reason"], "tool_use")
        self.assertEqual(message["usage"], {"input_tokens": 3, "output_tokens": 5})
        self.assertEqual(
            message["content"][0], {"type": "thinking", "thinking": "Because", "signature": ""}
        )
        self.assertEqual(message["content"][1], {"type": "text", "text": "Done"})
        self.assertEqual(message["content"][2]["input"], {"city": "Paris"})
        hidden = chat_to_message(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Stop here", "reasoning": "hidden"},
                    }
                ]
            },
            stop_sequences=["here"],
            include_thinking=False,
        )
        self.assertEqual(hidden["content"], [{"type": "text", "text": "Stop here"}])
        self.assertEqual(hidden["stop_reason"], "stop_sequence")
        self.assertEqual(hidden["stop_sequence"], "here")

    def test_stream_emits_the_messages_event_sequence(self):
        raw = io.BytesIO()
        stream = MessageEventStream(raw, input_tokens=4, stop_sequences=["END"])
        chunks = [
            {
                "id": "chatcmpl-abc",
                "model": "flint",
                "choices": [{"delta": {"reasoning": "Hmm"}, "finish_reason": None}],
            },
            {
                "choices": [{"delta": {"content": "Hello END"}, "finish_reason": None}],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {"choices": [], "usage": {"completion_tokens": 6}},
        ]
        stream.write(b": keepalive 1/2\n\n")
        for chunk in chunks:
            stream.write(f"data: {json.dumps(chunk)}\n\n".encode())
        stream.write(b"data: [DONE]\n\n")
        events = []
        for block in raw.getvalue().split(b"\n\n"):
            if not block or block.startswith(b":"):
                continue
            name, data = block.decode().split("\n", 1)
            events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
        names = [name for name, _ in events]
        self.assertEqual(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        self.assertEqual(events[0][1]["message"]["usage"]["input_tokens"], 4)
        self.assertEqual(events[2][1]["delta"], {"type": "thinking_delta", "thinking": "Hmm"})
        self.assertEqual(events[5][1]["delta"]["text"], "Hello END")
        self.assertEqual(events[7][1]["content_block"]["name"], "get_weather")
        self.assertEqual(events[8][1]["delta"]["partial_json"], '{"city":"Paris"}')
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(events[-2][1]["usage"]["output_tokens"], 6)
        self.assertTrue(raw.getvalue().startswith(b": keepalive 1/2\n\n"))

    def test_http_headers_are_written_before_events(self):
        raw = io.BytesIO()
        stream = MessageEventStream(raw, input_tokens=1)
        stream.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
        stream.write(
            b'data: {"id":"chatcmpl-a","model":"flint","choices":'
            b'[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
        )
        stream.write(b"data: [DONE]\n\n")
        body = raw.getvalue()
        self.assertLess(body.index(b"HTTP/1.1 200 OK"), body.index(b"event: message_start"))
        self.assertIn(b"event: content_block_delta", body)


if __name__ == "__main__":
    unittest.main()
