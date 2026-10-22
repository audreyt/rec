import json

_MAPPED = {
    "disable_parallel_tool_calls",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
}

_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def anthropic_error(status, message):
    kind = {
        400: "invalid_request_error",
        401: "authentication_error",
        404: "not_found_error",
        429: "rate_limit_error",
    }.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": message}}


def messages_to_chat(body):
    if not isinstance(body, dict):
        raise ValueError("Request should be a JSON dictionary")
    if "model" not in body:
        raise ValueError("model is required")
    if "max_tokens" not in body:
        raise ValueError("max_tokens is required")
    if "messages" not in body:
        raise ValueError("messages is required")
    messages = _messages(body["messages"])
    if "system" in body:
        messages.insert(0, {"role": "system", "content": _system(body["system"])})
    if not messages:
        raise ValueError("messages must contain text, thinking, or tool blocks")
    chat = {"model": body["model"], "max_tokens": body["max_tokens"], "messages": messages}
    for name in ("temperature", "top_p", "top_k"):
        if name in body:
            chat[name] = body[name]
    if "stop_sequences" in body:
        chat["stop"] = _stop_sequences(body["stop_sequences"])
    if "stream" in body:
        chat["stream"] = body["stream"]
    if body.get("stream") is True:
        chat["stream_options"] = {"include_usage": True}
    if "tools" in body:
        chat["tools"] = _tools(body["tools"])
    if "tool_choice" in body:
        chat["tool_choice"] = _tool_choice(body["tool_choice"])
    if "disable_parallel_tool_calls" in body:
        if type(body["disable_parallel_tool_calls"]) is not bool:
            raise ValueError("disable_parallel_tool_calls must be a boolean")
        if body["disable_parallel_tool_calls"]:
            raise ValueError("disable_parallel_tool_calls is not supported")
    return chat, set(body) - _MAPPED, _include_thinking(body.get("thinking"))


def chat_to_message(response, *, stop_sequences=(), include_thinking=True):
    choice = response["choices"][0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason") or "stop"
    text = message.get("content") or ""
    stop_reason, stop_sequence = _stop(finish, text, stop_sequences)
    usage = response.get("usage") or {}
    return {
        "id": _message_id(response.get("id")),
        "type": "message",
        "role": "assistant",
        "model": response.get("model", "flint"),
        "content": _blocks(message, include_thinking),
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


class MessageEventStream:
    def __init__(self, raw, *, input_tokens, stop_sequences=(), include_thinking=True):
        self.raw = raw
        self.input_tokens = input_tokens
        self.stop_sequences = tuple(stop_sequences)
        self.include_thinking = include_thinking
        self.pending = b""
        self.header_pending = b""
        self.headers_sent = False
        self.started = False
        self.message_id = "msg_0"
        self.model = "flint"
        self.index = 0
        self.open_block = None
        self.text = ""
        self.finish = None
        self.output_tokens = 0
        self.done = False

    def write(self, data):
        if not self.headers_sent and (self.header_pending or data.startswith(b"HTTP/")):
            self.header_pending += data
            marker = b"\r\n\r\n"
            if marker not in self.header_pending:
                return len(data)
            head, data = self.header_pending.split(marker, 1)
            self.header_pending = b""
            self.headers_sent = True
            self.raw.write(head + marker)
            self.raw.flush()
            if not data:
                return len(data)
        self.headers_sent = True
        if not self.pending and data[:1] == b"{":
            self.raw.write(_error_body(data))
            return len(data)
        self.pending += data
        while b"\n\n" in self.pending:
            event, self.pending = self.pending.split(b"\n\n", 1)
            self._event(event)
        return len(data)

    def flush(self):
        return self.raw.flush()

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def _event(self, event):
        if event.startswith(b":"):
            self.raw.write(event + b"\n\n")
            self.raw.flush()
            return
        if not event.startswith(b"data: "):
            return
        payload = event[6:]
        if payload == b"[DONE]":
            self._finish()
            return
        self._chunk(json.loads(payload))

    def _chunk(self, chunk):
        if not self.started:
            self.message_id = _message_id(chunk.get("id"))
            self.model = chunk.get("model", self.model)
        usage = chunk.get("usage")
        if usage and not chunk.get("choices"):
            self.output_tokens = usage.get("completion_tokens", 0)
            return
        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning")
        if reasoning and self.include_thinking:
            if self.open_block != "thinking":
                self._open("thinking", {"type": "thinking", "thinking": "", "signature": ""})
            self._delta({"type": "thinking_delta", "thinking": reasoning})
        content = delta.get("content")
        if content:
            if self.open_block != "text":
                self._open("text", {"type": "text", "text": ""})
            self.text += content
            self._delta({"type": "text_delta", "text": content})
        for call in delta.get("tool_calls") or []:
            function = call.get("function") or {}
            self._open(
                "tool",
                {
                    "type": "tool_use",
                    "id": call.get("id") or "",
                    "name": function.get("name") or "",
                    "input": {},
                },
            )
            self._delta(
                {"type": "input_json_delta", "partial_json": function.get("arguments") or "{}"}
            )
            self._close()
        if choice.get("finish_reason"):
            self.finish = choice["finish_reason"]

    def _open(self, kind, block):
        self._close()
        self._start()
        self._emit(
            "content_block_start",
            {"type": "content_block_start", "index": self.index, "content_block": block},
        )
        self.open_block = kind

    def _delta(self, delta):
        self._emit(
            "content_block_delta",
            {"type": "content_block_delta", "index": self.index, "delta": delta},
        )

    def _close(self):
        if self.open_block is None:
            return
        self._emit("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.index += 1
        self.open_block = None

    def _start(self):
        if self.started:
            return
        self.started = True
        self._emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            },
        )

    def _finish(self):
        if self.done:
            return
        self.done = True
        self._start()
        self._close()
        stop_reason, stop_sequence = _stop(self.finish or "stop", self.text, self.stop_sequences)
        self._emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": {"output_tokens": self.output_tokens},
            },
        )
        self._emit("message_stop", {"type": "message_stop"})

    def _emit(self, name, payload):
        body = f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
        self.raw.write(body)
        self.raw.flush()


def _error_body(data):
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return data
    if not isinstance(payload, dict) or "choices" in payload or "error" not in payload:
        return data
    message = payload["error"]
    if not isinstance(message, str):
        message = json.dumps(message)
    return json.dumps(anthropic_error(500, message)).encode()


def _message_id(value):
    return "msg_" + str(value or "0").removeprefix("chatcmpl-")


def _stop(finish, text, stop_sequences):
    if finish == "stop":
        for sequence in stop_sequences:
            if sequence and text.endswith(sequence):
                return "stop_sequence", sequence
    return _STOP_REASONS.get(finish, "end_turn"), None


def _blocks(message, include_thinking):
    blocks = []
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if include_thinking and reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": function.get("name"),
                "input": arguments,
            }
        )
    return blocks


def _include_thinking(thinking):
    if thinking is None:
        return True
    if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "disabled"):
        raise ValueError("thinking.type must be enabled or disabled")
    return thinking["type"] != "disabled"


def _system(system):
    if isinstance(system, str):
        return system
    if not isinstance(system, list) or not system:
        raise ValueError("system must be a string or a nonempty list of text blocks")
    return _text(system, "system")


def _text(blocks, where):
    parts = []
    for block in blocks:
        if (
            not isinstance(block, dict)
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
        ):
            raise ValueError(f"{where} blocks must be text")
        parts.append(block["text"])
    return "".join(parts)


def _stop_sequences(stops):
    if not isinstance(stops, list) or any(not isinstance(word, str) or not word for word in stops):
        raise ValueError("stop_sequences must be a list of nonempty strings")
    return stops


def _tool_choice(choice):
    if not isinstance(choice, dict):
        raise ValueError("tool_choice must be an object")
    kind = choice.get("type")
    if kind in ("auto", "none"):
        return kind
    if kind in ("any", "tool"):
        raise ValueError(
            "tool_choice currently supports auto and none; forced calls are not implemented"
        )
    raise ValueError("tool_choice type must be auto, none, any, or tool")


def _tools(tools):
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("each tool must be an object")
        if tool.get("type", "custom") != "custom":
            raise ValueError(f"Unsupported tool type {tool.get('type')!r}")
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a nonempty string")
        if not isinstance(schema, dict):
            raise ValueError(f"tool {name!r} requires an input_schema object")
        function = {"name": name, "parameters": schema}
        if "description" in tool:
            if not isinstance(tool["description"], str):
                raise ValueError(f"tool {name!r} description must be a string")
            function["description"] = tool["description"]
        converted.append({"type": "function", "function": function})
    return converted


def _messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    converted = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        if role == "user":
            converted.extend(_user(message.get("content")))
        elif role == "assistant":
            converted.append(_assistant(message.get("content")))
        else:
            raise ValueError("message roles must be user or assistant")
    return converted


def _user(content):
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        raise ValueError("user content must be a string or a list of blocks")
    converted = []
    text = []

    def flush():
        if text:
            converted.append({"role": "user", "content": "".join(text)})
            text.clear()

    for block in content:
        if not isinstance(block, dict):
            raise ValueError("content blocks must be objects")
        kind = block.get("type")
        if kind == "text":
            if not isinstance(block.get("text"), str):
                raise ValueError("text blocks require a string")
            text.append(block["text"])
        elif kind == "tool_result":
            flush()
            converted.append(_tool_result(block))
        else:
            raise ValueError(f"Unsupported user block type {kind!r}")
    flush()
    return converted


def _tool_result(block):
    tool_id = block.get("tool_use_id")
    if not isinstance(tool_id, str) or not tool_id:
        raise ValueError("tool_result requires a tool_use_id")
    content = block.get("content", "")
    if isinstance(content, list):
        content = _text(content, "tool_result")
    elif not isinstance(content, str):
        raise ValueError("tool_result content must be a string or text blocks")
    if "is_error" in block and type(block["is_error"]) is not bool:
        raise ValueError("tool_result is_error must be a boolean")
    if block.get("is_error"):
        content = "Error: " + content
    return {"role": "tool", "tool_call_id": tool_id, "content": content}


def _assistant(content):
    message = {"role": "assistant", "content": ""}
    if content is None:
        return message
    if isinstance(content, str):
        message["content"] = content
        return message
    if not isinstance(content, list):
        raise ValueError("assistant content must be a string or a list of blocks")
    texts = []
    thoughts = []
    calls = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("content blocks must be objects")
        kind = block.get("type")
        if kind == "text":
            if not isinstance(block.get("text"), str):
                raise ValueError("text blocks require a string")
            texts.append(block["text"])
        elif kind == "thinking":
            if not isinstance(block.get("thinking"), str):
                raise ValueError("thinking blocks require a string")
            thoughts.append(block["thinking"])
        elif kind == "tool_use":
            calls.append(_tool_use(block))
        else:
            raise ValueError(f"Unsupported assistant block type {kind!r}")
    message["content"] = "".join(texts)
    if thoughts:
        message["reasoning_content"] = "".join(thoughts)
    if calls:
        message["tool_calls"] = calls
    return message


def _tool_use(block):
    name = block.get("name")
    tool_id = block.get("id")
    arguments = block.get("input", {})
    if not isinstance(name, str) or not name:
        raise ValueError("tool_use requires a name")
    if not isinstance(tool_id, str) or not tool_id:
        raise ValueError("tool_use requires an id")
    if not isinstance(arguments, dict):
        raise ValueError("tool_use input must be an object")
    return {
        "id": tool_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }
