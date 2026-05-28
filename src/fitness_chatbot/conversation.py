"""In-memory conversation history."""

from __future__ import annotations


class Conversation:
    """Stores user/assistant turns; system prompt is added only at request time."""

    def __init__(self, max_turns: int = 20) -> None:
        self._messages: list[dict[str, str]] = []
        self._max_turns = max(1, max_turns)

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._messages)

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def clear(self) -> None:
        self._messages.clear()

    def pop_last_user(self) -> None:
        """Remove the last user message (e.g. after a failed or cancelled reply)."""
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()

    def _trim(self) -> None:
        max_messages = self._max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
