"""The normal form of a spelling: what two spellings of one entity have in common.

Case, hyphens and spaces, and a plural "s". Nothing that needs a dictionary or a model
(GLOSSARY: Нормальная форма); irregular forms are the synonym dictionary's job.
"""

import re
import unicodedata

_DASHES = str.maketrans(dict.fromkeys("-‐‑–—−", " "))
_WHITESPACE = re.compile(r"\s+")
_NOT_PLURAL_ENDINGS = ("ss", "us", "is", "as")


def normal_form(spelling: str) -> str:
    text = unicodedata.normalize("NFKC", spelling).casefold().translate(_DASHES)
    # One pass is not enough: rstrip runs before the final strip, so "Adaptation.-." keeps its
    # "." (it sits before a space) and a second call drops it. A stored mention must equal what
    # read_synonyms accepts as a key, so the function repeats until nothing changes. Every step
    # only shortens the text, so this terminates.
    previous = None
    while text != previous:
        previous = text
        text = _WHITESPACE.sub(" ", text).strip().rstrip(".,;:").strip()
        text = _singular_last_word(text)
    return text


def _singular_last_word(text: str) -> str:
    head, space, last = text.rpartition(" ")
    if len(last) > 3 and last.endswith("s") and not last.endswith(_NOT_PLURAL_ENDINGS):
        last = last[:-1]
    return head + space + last
