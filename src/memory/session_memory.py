from typing import List, Tuple


class ChatSessionMemory:
    """In-memory chat history for a single CLI session."""

    def __init__(self) -> None:
        self._messages: List[Tuple[str, str]] = []

    def append_user(self, text: str) -> None:
        self._messages.append(("user", text))

    def append_assistant(self, text: str) -> None:
        self._messages.append(("assistant", text))

    def messages(self) -> List[Tuple[str, str]]:
        return list(self._messages)
