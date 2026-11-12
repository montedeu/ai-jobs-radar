"""The Mention: what the model returns for one paper, and the schema that constrains it."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

EntityType = Literal["method", "model", "task", "dataset"]


class Mention(BaseModel):
    # Structured outputs require additionalProperties: false, which pydantic emits for "forbid".
    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    spelling: str  # as in the abstract, never rewritten (docs/adr/0002)
    expansion: str | None = None  # full form of an abbreviation, if the abstract spells it out

    @field_validator("spelling")
    @classmethod
    def _not_blank(cls, spelling: str) -> str:
        # The spelling itself stays as written; only a blank one names nothing.
        if not spelling.strip():
            raise ValueError("spelling is blank")
        return spelling


class PaperMentions(BaseModel):
    """The model's answer for one paper; the structured-output schema is built from it."""

    model_config = ConfigDict(extra="forbid")

    mentions: list[Mention]
