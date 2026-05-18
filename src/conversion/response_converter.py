import json
import re
import uuid
from html import unescape
from fastapi import HTTPException, Request
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest


def parse_function_arguments(arguments: str):
    """Parse function arguments, tolerating providers that concatenate JSON objects."""
    if not arguments:
        return {}

    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    values = []
    index = 0
    length = len(arguments)

    while index < length:
        while index < length and arguments[index].isspace():
            index += 1
        if index >= length:
            break

        try:
            value, next_index = decoder.raw_decode(arguments, index)
        except json.JSONDecodeError:
            return None

        values.append(value)
        index = next_index

    if not values:
        return None

    for value in reversed(values):
        if isinstance(value, dict) and value:
            return value

    return values[-1]


TRAILING_TEXTUAL_TOOL_CALL_RE = re.compile(
    r"(?P<prefix>.*?)\[Tool call id=(?P<id>\S+) name=(?P<name>\S+) input=(?P<input>.*)\]\s*$",
    re.DOTALL,
)
TRAILING_ANTML_TOOL_CALLS_RE = re.compile(
    r"(?P<prefix>.*?)[<\[]antml:function_calls>(?P<body>.*?)(?:</antml:function_calls>)?\s*$",
    re.DOTALL,
)
ANTML_INVOKE_RE = re.compile(
    r'<invoke\s+name="(?P<name>[^"]+)">\s*(?P<body>.*?)\s*</invoke>',
    re.DOTALL,
)
ANTML_PARAMETER_RE = re.compile(
    r'<parameter\s+name="(?P<name>[^"]+)">(?P<value>.*?)</parameter>',
    re.DOTALL,
)
INTERNAL_TOOL_RESULT_RE = re.compile(
    r"\s*\[Tool result for id=[^\]]+\]\s*.*",
    re.DOTALL,
)


def parse_textual_tool_call(text: str):
    """Parse providers that emit tool calls as text instead of structured tool_calls."""
    extracted = extract_textual_tool_call(text)
    if not extracted:
        return None

    return extracted["tool_call"]


def extract_textual_tool_call(text: str):
    """Extract a trailing textual tool call and hide echoed internal tool results."""
    match = TRAILING_TEXTUAL_TOOL_CALL_RE.match(text or "")
    if match:
        raw_arguments = match.group("input").strip()
        arguments = parse_function_arguments(raw_arguments)
        if arguments is None:
            arguments = {"raw_arguments": raw_arguments}

        prefix = strip_internal_tool_results(match.group("prefix")).strip()
        return {
            "prefix": prefix,
            "tool_call": {
                "id": match.group("id"),
                "name": match.group("name"),
                "input": arguments,
            },
        }

    antml_match = TRAILING_ANTML_TOOL_CALLS_RE.match(text or "")
    if not antml_match:
        return None

    tool_calls = parse_antml_tool_calls(antml_match.group("body"))
    if not tool_calls:
        return None

    prefix = strip_internal_tool_results(antml_match.group("prefix")).strip()
    return {
        "prefix": prefix,
        "tool_call": tool_calls[0],
    }


def parse_antml_tool_calls(body: str) -> list[dict]:
    tool_calls = []
    for invoke_match in ANTML_INVOKE_RE.finditer(body or ""):
        arguments = {}
        for parameter_match in ANTML_PARAMETER_RE.finditer(invoke_match.group("body")):
            name = parameter_match.group("name")
            raw_value = unescape(parameter_match.group("value")).strip()
            parsed_value = parse_function_arguments(raw_value)
            arguments[name] = raw_value if parsed_value is None else parsed_value

        tool_calls.append(
            {
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": invoke_match.group("name"),
                "input": arguments,
            }
        )

    return tool_calls


def strip_internal_tool_results(text: str) -> str:
    """Remove tool-result transcripts that are internal context, not user-facing text."""
    return INTERNAL_TOOL_RESULT_RE.sub("", text or "")


def build_tool_use_block(tool_call: dict) -> dict:
    return {
        "type": Constants.CONTENT_TOOL_USE,
        "id": tool_call["id"],
        "name": tool_call["name"],
        "input": tool_call["input"],
    }


def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Add text content
    text_content = message.get("content")
    if text_content is not None:
        extracted_tool_call = extract_textual_tool_call(text_content)
        if extracted_tool_call:
            if extracted_tool_call["prefix"]:
                content_blocks.append(
                    {"type": Constants.CONTENT_TEXT, "text": extracted_tool_call["prefix"]}
                )
            content_blocks.append(build_tool_use_block(extracted_tool_call["tool_call"]))
        else:
            visible_text = strip_internal_tool_results(text_content).strip()
            if visible_text:
                content_blocks.append({"type": Constants.CONTENT_TEXT, "text": visible_text})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            raw_arguments = function_data.get("arguments", "{}")
            arguments = parse_function_arguments(raw_arguments)
            if arguments is None:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)
    if any(block.get("type") == Constants.CONTENT_TOOL_USE for block in content_blocks):
        stop_reason = Constants.STOP_TOOL_USE

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get("completion_tokens", 0),
        },
    }

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Convert OpenAI streaming response to Claude streaming format."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # Process streaming chunks
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    text_buffer = []

    try:
        async for line in openai_stream:
            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse chunk: {chunk_data}, error: {e}")
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        text_buffer.append(delta["content"])

                    # Handle tool call deltas with improved incremental processing
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            # Initialize tool call tracking by index if not exists
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "json_sent": False,
                                    "claude_index": None,
                                    "started": False,
                                }

                            tool_call = current_tool_calls[tc_index]

                            # Update tool call ID if provided
                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            # Update function name and start content block if we have both id and name
                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]

                            # Start content block when we have complete initial data
                            if tool_call["id"] and tool_call["name"] and not tool_call["started"]:
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True

                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"

                            # Handle function arguments
                            if (
                                "arguments" in function_data
                                and tool_call["started"]
                                and function_data["arguments"] is not None
                            ):
                                tool_call["args_buffer"] += function_data["arguments"]

                                # Try to parse complete JSON and send delta when we have valid JSON
                                parsed_arguments = parse_function_arguments(
                                    tool_call["args_buffer"]
                                )
                                # If parsing succeeds and we haven't sent this JSON yet
                                if parsed_arguments is not None and not tool_call["json_sent"]:
                                    partial_json = json.dumps(parsed_arguments, ensure_ascii=False)
                                    yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_call['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': partial_json}}, ensure_ascii=False)}\n\n"
                                    tool_call["json_sent"] = True

                    # Handle finish reason
                    if finish_reason:
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN
                        break

    except Exception as e:
        # Handle any streaming errors gracefully
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return

    # Send final SSE events
    buffered_text = "".join(text_buffer)
    extracted_tool_call = extract_textual_tool_call(buffered_text)
    if extracted_tool_call:
        if extracted_tool_call["prefix"]:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': extracted_tool_call['prefix']}}, ensure_ascii=False)}\n\n"
        final_stop_reason = Constants.STOP_TOOL_USE
        tool_block_counter += 1
        claude_index = text_block_index + tool_block_counter
        tool_call = extracted_tool_call["tool_call"]
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"
        partial_json = json.dumps(tool_call["input"], ensure_ascii=False)
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': claude_index, 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': partial_json}}, ensure_ascii=False)}\n\n"
        current_tool_calls[claude_index] = {
            "started": True,
            "claude_index": claude_index,
        }
    elif buffered_text:
        visible_text = strip_internal_tool_results(buffered_text).strip()
        if visible_text:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': visible_text}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index}, ensure_ascii=False)}\n\n"

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']}, ensure_ascii=False)}\n\n"

    usage_data = {"input_tokens": 0, "output_tokens": 0}
    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data}, ensure_ascii=False)}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request,
    openai_client,
    request_id: str,
    on_complete=None,
):
    """Convert OpenAI streaming response to Claude streaming format with cancellation support."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # Process streaming chunks
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = {"input_tokens": 0, "output_tokens": 0}
    openai_model = None
    text_buffer = []

    try:
        async for line in openai_stream:
            # Check if client disconnected
            if await http_request.is_disconnected():
                logger.info(f"Client disconnected, cancelling request {request_id}")
                openai_client.cancel_request(request_id)
                break

            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        # logger.info(f"OpenAI chunk: {chunk}")
                        if chunk.get("model"):
                            openai_model = chunk.get("model")
                        usage = chunk.get("usage", None)
                        if usage:
                            cache_read_input_tokens = 0
                            prompt_tokens_details = usage.get("prompt_tokens_details", {})
                            if prompt_tokens_details:
                                cache_read_input_tokens = (
                                    prompt_tokens_details.get("cached_tokens", 0) or 0
                                )
                            usage_data = {
                                "input_tokens": usage.get("prompt_tokens", 0),
                                "output_tokens": usage.get("completion_tokens", 0),
                                "cache_read_input_tokens": cache_read_input_tokens,
                            }
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse chunk: {chunk_data}, error: {e}")
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        text_buffer.append(delta["content"])

                    # Handle tool call deltas with improved incremental processing
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            # Initialize tool call tracking by index if not exists
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "json_sent": False,
                                    "claude_index": None,
                                    "started": False,
                                }

                            tool_call = current_tool_calls[tc_index]

                            # Update tool call ID if provided
                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            # Update function name and start content block if we have both id and name
                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]

                            # Start content block when we have complete initial data
                            if tool_call["id"] and tool_call["name"] and not tool_call["started"]:
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True

                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"

                            # Handle function arguments
                            if (
                                "arguments" in function_data
                                and tool_call["started"]
                                and function_data["arguments"] is not None
                            ):
                                tool_call["args_buffer"] += function_data["arguments"]

                                # Try to parse complete JSON and send delta when we have valid JSON
                                try:
                                    json.loads(tool_call["args_buffer"])
                                    # If parsing succeeds and we haven't sent this JSON yet
                                    if not tool_call["json_sent"]:
                                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_call['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_call['args_buffer']}}, ensure_ascii=False)}\n\n"
                                        tool_call["json_sent"] = True
                                except json.JSONDecodeError:
                                    # JSON is incomplete, continue accumulating
                                    pass

                    # Handle finish reason
                    if finish_reason:
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN

    except HTTPException as e:
        # Handle cancellation
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            error_event = {
                "type": "error",
                "error": {
                    "type": "cancelled",
                    "message": "Request was cancelled by client",
                },
            }
            yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            return

        logger.error(f"Upstream streaming request failed: {e.detail}")
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e.detail)},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return
    except Exception as e:
        # Handle any streaming errors gracefully
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return

    # Send final SSE events
    buffered_text = "".join(text_buffer)
    extracted_tool_call = extract_textual_tool_call(buffered_text)
    if extracted_tool_call:
        if extracted_tool_call["prefix"]:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': extracted_tool_call['prefix']}}, ensure_ascii=False)}\n\n"
        final_stop_reason = Constants.STOP_TOOL_USE
        tool_block_counter += 1
        claude_index = text_block_index + tool_block_counter
        tool_call = extracted_tool_call["tool_call"]
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"
        partial_json = json.dumps(tool_call["input"], ensure_ascii=False)
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': claude_index, 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': partial_json}}, ensure_ascii=False)}\n\n"
        current_tool_calls[claude_index] = {
            "started": True,
            "claude_index": claude_index,
        }
    elif buffered_text:
        visible_text = strip_internal_tool_results(buffered_text).strip()
        if visible_text:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': visible_text}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index}, ensure_ascii=False)}\n\n"

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']}, ensure_ascii=False)}\n\n"

    if on_complete:
        on_complete(usage_data, final_stop_reason, openai_model)

    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data}, ensure_ascii=False)}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"
