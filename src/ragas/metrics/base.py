"""Metric base class."""

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragas.embeddings import Embeddings
from ragas.llm import StructuredLLM
from ragas.schema import SingleTurnSample


@dataclass
class Metric(ABC):
    name: str
    required_columns: t.Set[str]
    llm: t.Optional[StructuredLLM] = None
    embeddings: t.Optional[Embeddings] = None

    @abstractmethod
    async def single_turn_ascore(self, sample: SingleTurnSample) -> float:
        """Score one sample; returns 0..1 or NaN."""
