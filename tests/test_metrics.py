"""Behavior tests for the agno-based ragas core: metrics score correctly."""

import numpy as np
import pytest

from ragas.embeddings import Embeddings
from ragas.llm import StructuredLLM
from ragas.metrics import (
    ContextPrecision,
    Faithfulness,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.schema import EvaluationDataset, SingleTurnSample


class FakeLLM(StructuredLLM):
    """Returns canned outputs keyed by output schema name."""

    def __init__(self, responses: dict):
        super().__init__(model=None)
        self.responses = responses

    async def generate(self, instruction, input_data, output_schema, examples=()):
        return self.responses[output_schema.__name__]


class FakeEmbeddings(Embeddings):
    def __init__(self, vectors: dict):
        self.vectors = vectors

    def embed_query(self, text: str):
        return self.vectors[text]

    def embed_documents(self, texts):
        return [self.vectors[t] for t in texts]


def test_faithfulness():
    from ragas.metrics._faithfulness import (
        NLIStatementOutput,
        StatementGeneratorOutput,
    )

    llm = FakeLLM(
        {
            "StatementGeneratorOutput": StatementGeneratorOutput(
                statements=["Einstein was born in Germany.", "Einstein liked music."]
            ),
            "NLIStatementOutput": NLIStatementOutput(
                statements=[
                    {
                        "statement": "Einstein was born in Germany.",
                        "verdict": 1,
                        "reason": "",
                    },
                    {"statement": "Einstein liked music.", "verdict": 0, "reason": ""},
                ]
            ),
        }
    )
    m = Faithfulness(llm=llm)
    s = SingleTurnSample(
        user_input="Who was Einstein?",
        response="Einstein was born in Germany. He liked music.",
        retrieved_contexts=["Einstein was born in Germany."],
    )
    score = __import__("asyncio").run(m.single_turn_ascore(s))
    assert score == 0.5


def test_faithfulness_no_statements_nan():
    from ragas.metrics._faithfulness import StatementGeneratorOutput

    llm = FakeLLM({"StatementGeneratorOutput": StatementGeneratorOutput(statements=[])})
    m = Faithfulness(llm=llm)
    s = SingleTurnSample(user_input="q", response="a", retrieved_contexts=["c"])
    score = __import__("asyncio").run(m.single_turn_ascore(s))
    assert np.isnan(score)


def test_context_recall():
    from ragas.metrics._context_recall import ContextRecallClassifications

    llm = FakeLLM(
        {
            "ContextRecallClassifications": ContextRecallClassifications(
                classifications=[
                    {"statement": "s1", "attributed": 1, "reason": ""},
                    {"statement": "s2", "attributed": 0, "reason": ""},
                    {"statement": "s3", "attributed": 1, "reason": ""},
                ]
            )
        }
    )
    m = LLMContextRecall(llm=llm)
    s = SingleTurnSample(
        user_input="q", reference="s1. s2. s3.", retrieved_contexts=["ctx"]
    )
    score = __import__("asyncio").run(m.single_turn_ascore(s))
    assert score == pytest.approx(2 / 3)


def test_context_precision():
    from ragas.metrics._context_precision import Verification

    llm = FakeLLM(
        {
            # one call per retrieved context; both relevant
            "Verification": Verification(reason="", verdict=1),
        }
    )
    m = ContextPrecision(llm=llm)
    s = SingleTurnSample(user_input="q", reference="a", retrieved_contexts=["c1", "c2"])
    score = __import__("asyncio").run(m.single_turn_ascore(s))
    # both verdicts 1 -> AP = 1.0
    assert score == pytest.approx(1.0)


def test_answer_relevancy():
    from ragas.metrics._answer_relevance import ResponseRelevanceOutput

    llm = FakeLLM(
        {
            "ResponseRelevanceOutput": ResponseRelevanceOutput(
                question="Where was Einstein born?", noncommittal=0
            )
        }
    )
    emb = FakeEmbeddings(
        {
            "Who was Einstein?": [1.0, 0.0],
            "Where was Einstein born?": [0.9, 0.1],
        }
    )
    m = ResponseRelevancy(llm=llm, embeddings=emb, strictness=1)
    s = SingleTurnSample(user_input="Who was Einstein?", response="Born in Germany.")
    score = __import__("asyncio").run(m.single_turn_ascore(s))
    assert 0.9 < score <= 1.0


def test_answer_relevancy_noncommittal_zero():
    from ragas.metrics._answer_relevance import ResponseRelevanceOutput

    llm = FakeLLM(
        {
            "ResponseRelevanceOutput": ResponseRelevanceOutput(
                question="whatever", noncommittal=1
            )
        }
    )
    emb = FakeEmbeddings({"Who was Einstein?": [1.0, 0.0], "whatever": [1.0, 0.0]})
    m = ResponseRelevancy(llm=llm, embeddings=emb, strictness=1)
    s = SingleTurnSample(user_input="Who was Einstein?", response="I don't know.")
    score = __import__("asyncio").run(m.single_turn_ascore(s))
    assert score == 0.0


def test_evaluate_dataset():
    from ragas.evaluation import evaluate
    from ragas.metrics._faithfulness import (
        NLIStatementOutput,
        StatementGeneratorOutput,
    )

    llm = FakeLLM(
        {
            "StatementGeneratorOutput": StatementGeneratorOutput(statements=["s."]),
            "NLIStatementOutput": NLIStatementOutput(
                statements=[{"statement": "s.", "verdict": 1, "reason": ""}]
            ),
        }
    )
    ds = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="q", response="a", retrieved_contexts=["c"])
        ]
    )
    result = evaluate(ds, [Faithfulness(llm=llm)], llm=llm)
    assert result.scores["faithfulness"] == pytest.approx(1.0)


def test_required_columns_validated():
    from ragas.evaluation import evaluate
    from ragas.metrics._faithfulness import Faithfulness

    ds = EvaluationDataset(samples=[SingleTurnSample(user_input="q", response="a")])
    with pytest.raises(ValueError, match="retrieved_contexts"):
        evaluate(ds, [Faithfulness(llm=FakeLLM({}))], llm=FakeLLM({}))
