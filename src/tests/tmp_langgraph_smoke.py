from agents.dialog_agent import build_dialog_runtime
from config.settings import load_settings
from memory.session_memory import ChatSessionMemory


def main() -> None:
    settings = load_settings()
    runner, runtime = build_dialog_runtime(settings)
    memory = ChatSessionMemory()
    memory.append_user("现在几点了？请使用工具回答。")

    result = runner.invoke({"messages": memory.messages()})
    messages = result.get("messages", [])
    answer = getattr(messages[-1], "content", "") if messages else ""

    print("RUNTIME", runtime)
    print("ANSWER", answer)
    print("LANGGRAPH_SMOKE_OK")


if __name__ == "__main__":
    main()

