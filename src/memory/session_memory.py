from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

class ChatSessionMemory:
    """In-memory chat history for a single CLI session."""

    def __init__(self) -> None:
        self._messages: List[BaseMessage] = []

    def append_user(self, text: str) -> None:
        self._messages.append(HumanMessage(content=text))

    def append_assistant(self, text: str) -> None:
        self._messages.append(AIMessage(content=text))

    def messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages.clear()

    def serialize(self) -> List[dict]:
        payload: List[dict] = []
        for msg in self._messages:
            payload.append({"type": msg.type, "content": msg.content})
        return payload
