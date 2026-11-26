"""Data shapes for the agno-based ragas core."""

import typing as t

from pydantic import BaseModel, Field


class SingleTurnSample(BaseModel):
    """One QA row with optional retrieved/reference contexts."""

    user_input: str = Field(description="The question")
    response: t.Optional[str] = Field(default=None, description="The generated answer")
    reference: t.Optional[str] = Field(default=None, description="The expected answer")
    retrieved_contexts: t.Optional[t.List[str]] = Field(
        default=None, description="Contexts retrieved by the RAG pipeline"
    )
    reference_contexts: t.Optional[t.List[str]] = Field(
        default=None, description="Golden contexts for this question"
    )


class EvaluationDataset:
    """A list of samples plus column validation against metrics."""

    def __init__(self, samples: t.List[SingleTurnSample]):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def validate_columns(self, required: t.Set[str]) -> None:
        sample = self.samples[0]
        data = sample.model_dump()
        missing = {
            col
            for col in required
            if data.get(col) is None
        }
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}. "
                f"Fill them on the first sample to proceed."
            )


class EvaluationResult(BaseModel):
    """Aggregated evaluation output."""

    scores: t.Dict[str, float] = Field(description="Mean score per metric")
    scores_per_row: t.List[t.Dict[str, t.Optional[float]]] = Field(
        description="Per-sample metric scores"
    )
