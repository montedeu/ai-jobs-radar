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
    # A leading full stop is part of the name (".NET" is not "net"), so the start loses only
    # spaces and ",;:", and the end loses the full stop too. Dashes become spaces first, so
    # "Adaptation.-." reaches "Adaptation. ." and the one rstrip takes the whole trailing run.
    # Stripping in two steps left a "." there, and a second call dropped it; a stored mention
    # has to equal the key that read_synonyms accepts, so the function must reach its own fixed
    # point in one pass.
    text = _WHITESPACE.sub(" ", text).lstrip(" ,;:").rstrip(" .,;:")
    return _singular_last_word(text)


def _singular_last_word(text: str) -> str:
    # After a letter or a digit: "GPT4s" and "A100s" are plurals, "glasses.s" is not. The word
    # then ends in a letter or digit, not in "s" after one, so a second call changes nothing.
    head, space, last = text.rpartition(" ")
    if (
        len(last) > 3
        and last.endswith("s")
        and last[-2].isalnum()
        and not last[:-1].isdigit()  # "1990s" is a decade, not a plural of "1990"
        and not last.endswith(_NOT_PLURAL_ENDINGS)
    ):
        last = last[:-1]
    return head + space + last
