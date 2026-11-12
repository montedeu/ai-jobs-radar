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
    # Dashes become spaces before the edges are stripped, and the edges are stripped with one
    # call that takes spaces and punctuation alike. Stripping punctuation and then spaces in
    # two steps left "Adaptation.-." with its "." (a dash sat between them), so a second call
    # dropped it: the function must reach its own fixed point in one pass, because a stored
    # mention has to equal the key that read_synonyms accepts.
    text = _WHITESPACE.sub(" ", text).strip(" .,;:")
    return _singular_last_word(text)


def _singular_last_word(text: str) -> str:
    # Only after a letter: "glasses.s" is not a plural. A word that ends in "s" after a letter
    # loses one "s" and then ends in a letter, so a second call changes nothing.
    head, space, last = text.rpartition(" ")
    if (
        len(last) > 3
        and last.endswith("s")
        and last[-2].isalpha()
        and not last.endswith(_NOT_PLURAL_ENDINGS)
    ):
        last = last[:-1]
    return head + space + last
