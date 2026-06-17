import json
import logging
import uuid

from agents.dialog_agent import build_dialog_runtime
from config.logging_setup import setup_logging
from config.settings import load_settings
from db.connection import init_postgres_pool
from db.conversations import create_conversation, list_conversations, update_conversation, update_title
from db.messages import save_message, list_messages
from rag.core.bootstrap import bootstrap_rag


logger = logging.getLogger("chat.cli")

_INTERNAL_STREAM_NODES = {
    "supervisor",
    "image_input_guardrail",
    "image_caption",
    "input_guardrail",
    "triage",
    "decompose",
    "worker",
}
_FINAL_STREAM_NODES = {"single", "image", "synthesize", "output_guardrail"}

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


def _stream_langgraph(dialog_runner, runtime: str, session_id: str, message: str) -> tuple[str, list[dict]]:
    latest_answer = ""
    seen_answer = ""
    collected_tool_calls: list[dict] = []

    for update in dialog_runner.stream(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": session_id}},
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue
        for node_name, node_state in update.items():
            if node_name in _INTERNAL_STREAM_NODES:
                continue
            if runtime == "langgraph-swarm" and node_name not in _FINAL_STREAM_NODES:
                continue
            if not isinstance(node_state, dict):
                continue
            messages = node_state.get("messages", [])
            for msg in messages:
                collected_tool_calls.extend(_extract_tool_calls_from_message(msg))
                if getattr(msg, "type", "") == "ai":
                    text = _extract_text_from_content(getattr(msg, "content", "")).strip()
                    if text:
                        latest_answer = text
                        if text != seen_answer:
                            print(f"Assistant: {text}")
                            seen_answer = text

    return latest_answer, collected_tool_calls


def run_chat() -> None:
    setup_logging()
    settings = load_settings()

    if settings.postgres_uri:
        init_postgres_pool(settings.postgres_uri)

    if settings.rag_enabled:
        try:
            ok, reason = bootstrap_rag(settings)
            if ok:
                logger.info("rag_bootstrap_ok reason=%s", reason)
            else:
                logger.warning("rag_bootstrap_failed reason=%s", reason)
        except Exception as exc:
            logger.warning("rag_bootstrap_failed reason=%s", exc)
    else:
        logger.info("rag_bootstrap_skipped reason=disabled")

    dialog_runner, runtime = build_dialog_runtime(settings)

    session_id = str(uuid.uuid4())
    create_conversation(session_id)

    print(f"Chat runtime: {runtime}. Session: {session_id[:8]}...")
    print("Commands: /new (new chat) | /list (history) | /quit")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if user_input == "/new":
            session_id = str(uuid.uuid4())
            create_conversation(session_id)
            print(f"New session: {session_id[:8]}...")
            continue
        if user_input == "/list":
            for c in list_conversations(limit=10):
                print(f"{c['session_id'][:8]}  {c['title'] or '(untitled)'}  {c['updated_at']}")
            continue
        if not user_input:
            continue

        try:
            save_message(session_id, "user", user_input)
            answer = ""
            tool_calls: list[dict] = []
            if runtime.startswith("langgraph") and settings.agent_streaming:
                answer, tool_calls = _stream_langgraph(dialog_runner, runtime, session_id, user_input)
                if not answer:
                    result = dialog_runner.invoke(
                        {"messages": [("user", user_input)]},
                        config={"configurable": {"thread_id": session_id}},
                    )
                    tool_calls = _extract_tool_calls(result)
                    answer = _extract_text_from_result(result)
            else:
                result = dialog_runner.invoke(
                    {"messages": [("user", user_input)]},
                    config={"configurable": {"thread_id": session_id}},
                )
                tool_calls = _extract_tool_calls(result)
                answer = _extract_text_from_result(result)

            save_message(session_id, "assistant", answer)
            update_conversation(session_id)

            msgs = list_messages(session_id, limit=2)
            if len(msgs) <= 2:
                title = user_input[:30].strip()
                if title:
                    update_title(session_id, title)

            _log_tool_calls(tool_calls)
            logger.info("[CHAT_RESULT] answer=%s", answer)
            if not (runtime.startswith("langgraph") and settings.agent_streaming):
                print(f"Assistant: {answer}")
        except Exception as exc:
            logger.exception("[CHAT_ERROR] %s", exc)
            print(f"Error: {exc}")


if __name__ == "__main__":
    run_chat()
