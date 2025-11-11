"""
Turning a token stream into speakable sentences.

This is the single most important piece of latency work in a voice agent, and
it is four lines of regex. A model that takes 1.4 s to finish a three-sentence
answer has usually finished the *first* sentence in 300 ms — so waiting for the
whole response before synthesising anything throws away a second of silence on
every turn, for nothing.

Segmenting the stream at sentence boundaries lets synthesis and playback start
while the model is still writing. The speaker becomes the bottleneck instead of
the provider, which is the right way round: speech plays at a fixed rate, so
once the first sentence is out, the rest has the whole of its playback duration
to arrive.

It lives here rather than in the CLI client because both the CLI and the
push-to-talk handler need it, and because it is a property of the pipeline, not
of any one front end.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator

#: The shortest text that ends in terminal punctuation followed by whitespace.
#: Non-greedy so a paragraph yields its sentences one at a time rather than in
#: one lump at the end.
_SENTENCE_RE = re.compile(r"([\s\S]*?[.!?])\s")


def split_sentences(
    text_stream: Iterable[str],
    stop_flag: Callable[[], bool] = lambda: False,
    on_partial: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """
    Yield complete sentences from a stream of text fragments.

    Fragments are forwarded to `on_partial` as they arrive, so a UI can show
    the reply being written while the same stream is being spoken. A true
    `stop_flag` abandons the remainder — that is barge-in: the caller started
    talking, so the rest of the sentence is no longer wanted.
    """
    buffer = ""
    for token in text_stream:
        if stop_flag():
            return
        if on_partial:
            on_partial(token)
        buffer += token
        while True:
            match = _SENTENCE_RE.search(buffer)
            if not match:
                break
            sentence = match.group(1)
            yield sentence
            buffer = buffer[len(sentence) :].lstrip()
    if not stop_flag() and buffer.strip():
        yield buffer.strip()
