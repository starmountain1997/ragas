"""Data shapes for the agno-based ragas core."""

from pydantic import BaseModel, Field


class SingleTurnSample(BaseModel):
    """One QA row with optional retrieved/reference contexts."""

    user_input: str = Field(description="The question")
    response: str | None = Field(default=None, description="The generated answer")
    reference: str | None = Field(default=None, description="The expected answer")
    retrieved_contexts: list[str] | None = Field(
        default=None, description="Contexts retrieved by the RAG pipeline"
    )
    reference_contexts: list[str] | None = Field(
        default=None, description="Golden contexts for this question"
    )


class EvaluationDataset:
    """A list of samples plus column validation against metrics."""

    def __init__(self, samples: list[SingleTurnSample]):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def validate_columns(self, required: set[str]) -> None:
        sample = self.samples[0]
        data = sample.model_dump()
        missing = {col for col in required if data.get(col) is None}
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}. "
                f"Fill them on the first sample to proceed."
            )


class EvaluationResult(BaseModel):
    """Aggregated evaluation output."""

    scores: dict[str, float] = Field(description="Mean score per metric")
    scores_per_row: list[dict[str, float | None]] = Field(
        description="Per-sample metric scores"
    )
