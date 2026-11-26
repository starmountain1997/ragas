"""evaluate() entry point."""

import asyncio
import typing as t

import numpy as np
from tqdm.auto import tqdm

from ragas.schema import EvaluationDataset, EvaluationResult
from ragas.metrics.base import Metric


async def aevaluate(
    dataset: EvaluationDataset,
    metrics: t.Sequence[Metric],
    llm: t.Optional[t.Any] = None,
    embeddings: t.Optional[t.Any] = None,
    show_progress: bool = True,
) -> EvaluationResult:
    """Score every sample with every metric, concurrently, and aggregate."""

    for metric in metrics:
        if metric.llm is None:
            metric.llm = llm
        if metric.embeddings is None:
            metric.embeddings = embeddings
        dataset.validate_columns(metric.required_columns)

    async def score_one(metric: Metric, sample) -> float:
        try:
            return await metric.single_turn_ascore(sample)
        except Exception:
            return np.nan

    tasks = [
        [score_one(metric, sample) for metric in metrics] for sample in dataset
    ]
    pbar = tqdm(total=len(tasks) * len(metrics), disable=not show_progress)

    async def track(coro):
        result = await coro
        pbar.update(1)
        return result

    results = await asyncio.gather(
        *[asyncio.gather(*[track(c) for c in row]) for row in tasks]
    )
    pbar.close()

    names = [m.name for m in metrics]
    scores_per_row = [dict(zip(names, row)) for row in results]
    means = {}
    for name in names:
        vals = [row[name] for row in scores_per_row]
        vals = [v for v in vals if not np.isnan(v)]
        means[name] = float(np.mean(vals)) if vals else np.nan
    return EvaluationResult(scores=means, scores_per_row=scores_per_row)


def evaluate(
    dataset: EvaluationDataset,
    metrics: t.Sequence[Metric],
    llm: t.Optional[t.Any] = None,
    embeddings: t.Optional[t.Any] = None,
    show_progress: bool = True,
) -> EvaluationResult:
    return asyncio.run(
        aevaluate(dataset, metrics, llm=llm, embeddings=embeddings, show_progress=show_progress)
    )
