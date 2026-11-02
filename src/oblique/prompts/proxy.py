"""proxy-sequence-v1: the fixed text of a sequential set in the OpenAI-compatible proxy.

A client's request is any chat, so earlier answers are plain text, not idea cards. The card keeps
its frame `inject-v1`. Both go after the client's last message, so all the calls of a set share the
client's text as a prefix for the provider's prompt cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from oblique.prompts.base import Prompt
from oblique.prompts.ideation import INJECT_V1

PREVIOUS = """Answers already given to this request:

{answers}

The new answer must be distinct from these.

"""

ANSWER = """Answer {number}:
{answer}"""

PROXY_SEQUENCE_V1 = Prompt("proxy-sequence-v1", "", PREVIOUS, extra=(("answer", ANSWER),))


def oblique_messages(
    messages: Sequence[dict[str, Any]], previous: Sequence[str], card: str
) -> list[dict[str, Any]]:
    """The client's chat with the earlier answers and the card added to its last message."""
    block = ""
    if previous:
        answers = "\n\n".join(
            ANSWER.format(number=i, answer=text) for i, text in enumerate(previous, 1)
        )
        block = PREVIOUS.format(answers=answers)
    added = "\n\n" + block + INJECT_V1.user.format(card=card).rstrip()
    last = dict(messages[-1])
    content = last["content"]
    if isinstance(content, str):
        last["content"] = content + added
    else:
        last["content"] = [*content, {"type": "text", "text": added}]
    return [*messages[:-1], last]
