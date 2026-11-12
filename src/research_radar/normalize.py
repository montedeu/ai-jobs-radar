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
    text = _WHITESPACE.sub(" ", text).strip().rstrip(".,;:").strip()
    return _singular_last_word(text)


def _singular_last_word(text: str) -> str:
    head, space, last = text.rpartition(" ")
    if len(last) > 3 and last.endswith("s") and not last.endswith(_NOT_PLURAL_ENDINGS):
        last = last[:-1]
    return head + space + last
