import json
import logging

from agents.dialog_agent import build_dialog_agent
from config.logging_setup import setup_logging
from config.settings import load_settings
from memory.session_memory import ChatSessionMemory


logger = logging.getLogger("chat.cli")


def _extract_tool_calls(result: dict) -> list[dict]:
    calls: list[dict] = []
    for message in result.get("messages", []):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and hasattr(message, "additional_kwargs"):
            tool_calls = message.additional_kwargs.get("tool_calls")
        if tool_calls:
            for call in tool_calls:
                if isinstance(call, dict):
                    calls.append(call)
    return calls


def _extract_text_from_result(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response from model."

    last = messages[-1]
    content = getattr(last, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        merged = "".join(text_parts).strip()
        return merged or str(content)

    return str(content)


def run_chat() -> None:
    setup_logging()
    settings = load_settings()
    dialog_agent = build_dialog_agent(settings)
    memory = ChatSessionMemory()

    print("LangChain Agent started. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        try:
            memory.append_user(user_input)
            result = dialog_agent.invoke({"messages": memory.messages()})
            tool_calls = _extract_tool_calls(result)
            if tool_calls:
                for call in tool_calls:
                    name = call.get("name", "unknown")
                    args = call.get("args", call.get("arguments", {}))
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    logger.info("[TOOL_USED] name=%s args=%s", name, args)
            else:
                logger.info("[TOOL_SKIP] no tool call in this turn")
            answer = _extract_text_from_result(result)
            memory.append_assistant(answer)
            logger.info("[CHAT_RESULT] answer=%s", answer)
            print(f"Assistant: {answer}")
        except Exception as exc:
            logger.exception("[CHAT_ERROR] %s", exc)
            print(f"Error: {exc}")


if __name__ == "__main__":
    run_chat()
