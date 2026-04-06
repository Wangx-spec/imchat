import json
import logging

from agents.dialog_agent import build_dialog_runtime
from config.logging_setup import setup_logging
from config.settings import load_settings
from memory.session_memory import ChatSessionMemory


logger = logging.getLogger("chat.cli")


def _extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts).strip()

    return str(content)


def _extract_tool_calls_from_message(message) -> list[dict]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls and hasattr(message, "additional_kwargs"):
        tool_calls = message.additional_kwargs.get("tool_calls")
    if isinstance(tool_calls, list):
        return [call for call in tool_calls if isinstance(call, dict)]
    return []


def _extract_tool_calls(result: dict) -> list[dict]:
    calls: list[dict] = []
    for message in result.get("messages", []):
        calls.extend(_extract_tool_calls_from_message(message))
    return calls


def _extract_text_from_result(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response from model."

    last = messages[-1]
    content = getattr(last, "content", "")
    merged = _extract_text_from_content(content)
    return merged or str(content)


def _log_tool_calls(tool_calls: list[dict]) -> None:
    if tool_calls:
        for call in tool_calls:
            name = call.get("name", "unknown")
            args = call.get("args", call.get("arguments", {}))
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            logger.info("[TOOL_USED] name=%s args=%s", name, args)
    else:
        logger.info("[TOOL_SKIP] no tool call in this turn")


def _stream_langgraph(dialog_runner, history) -> tuple[str, list[dict]]:
    latest_answer = ""
    seen_answer = ""
    collected_tool_calls: list[dict] = []

    for update in dialog_runner.stream({"messages": history}, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_state in update.values():
            if not isinstance(node_state, dict):
                continue
            messages = node_state.get("messages", [])
            for message in messages:
                collected_tool_calls.extend(_extract_tool_calls_from_message(message))
                if getattr(message, "type", "") == "ai":
                    text = _extract_text_from_content(getattr(message, "content", "")).strip()
                    if text:
                        latest_answer = text
                        if text != seen_answer:
                            print(f"Assistant: {text}")
                            seen_answer = text

    return latest_answer, collected_tool_calls


def run_chat() -> None:
    setup_logging()
    settings = load_settings()
    dialog_runner, runtime = build_dialog_runtime(settings)
    memory = ChatSessionMemory()

    print(f"Chat runtime: {runtime}. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        try:
            memory.append_user(user_input)
            history = memory.messages()

            answer = ""
            tool_calls: list[dict] = []
            if runtime == "langgraph" and settings.agent_streaming:
                answer, tool_calls = _stream_langgraph(dialog_runner, history)
                if not answer:
                    # Safety fallback for providers that do not return updates content.
                    result = dialog_runner.invoke({"messages": history})
                    tool_calls = _extract_tool_calls(result)
                    answer = _extract_text_from_result(result)
            else:
                result = dialog_runner.invoke({"messages": history})
                tool_calls = _extract_tool_calls(result)
                answer = _extract_text_from_result(result)

            _log_tool_calls(tool_calls)
            memory.append_assistant(answer)
            logger.info("[CHAT_RESULT] answer=%s", answer)
            if runtime != "langgraph" or not settings.agent_streaming:
                print(f"Assistant: {answer}")
        except Exception as exc:
            logger.exception("[CHAT_ERROR] %s", exc)
            print(f"Error: {exc}")


if __name__ == "__main__":
    run_chat()
