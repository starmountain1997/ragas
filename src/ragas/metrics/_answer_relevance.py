import typing as t

import numpy as np
from pydantic import BaseModel, Field

from ragas.metrics.base import Metric
from ragas.schema import SingleTurnSample


class ResponseRelevanceInput(BaseModel):
    response: str


class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int


INSTRUCTION = (
    "Generate a question for the given answer and Identify if answer is "
    "noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if "
    'the answer is committal. A noncommittal answer is one that is evasive, vague, '
    'or ambiguous. For example, "I don\'t know" or "I\'m not sure" are noncommittal '
    "answers"
)
EXAMPLES = [
    (
        ResponseRelevanceInput(response="Albert Einstein was born in Germany."),
        ResponseRelevanceOutput(question="Where was Albert Einstein born?", noncommittal=0),
    ),
    (
        ResponseRelevanceInput(
            response="I don't know about the groundbreaking feature of the smartphone "
            "invented in 2023 as am unaware of information beyond 2022."
        ),
        ResponseRelevanceOutput(
            question="What was the groundbreaking feature of the smartphone invented "
            "in 2023?",
            noncommittal=1,
        ),
    ),
]


class ResponseRelevancy(Metric):
    def __init__(
        self,
        llm=None,
        embeddings=None,
        name: str = "answer_relevancy",
        strictness: int = 3,
    ):
        super().__init__(
            name=name,
            required_columns={"user_input", "response"},
            llm=llm,
            embeddings=embeddings,
        )
        self.strictness = strictness

    async def single_turn_ascore(self, sample: SingleTurnSample) -> float:
        assert self.llm is not None, "LLM is not set"
        assert self.embeddings is not None, "embeddings must be set"

        # one question per sample; strictness repeats are rolled into one call
        answers = []
        for _ in range(self.strictness):
            answer = await self.llm.generate(
                INSTRUCTION,
                ResponseRelevanceInput(response=sample.response or ""),
                ResponseRelevanceOutput,
                EXAMPLES,
            )
            answers.append(answer)

        gen_questions = [a.question for a in answers]
        if all(q == "" for q in gen_questions):
            return np.nan

        all_noncommittal = all(a.noncommittal for a in answers)
        if all_noncommittal:
            return 0.0

        question_vec = np.asarray(
            self.embeddings.embed_query(sample.user_input)
        ).reshape(1, -1)
        gen_vec = np.asarray(
            self.embeddings.embed_documents(gen_questions)
        ).reshape(len(gen_questions), -1)
        norm = np.linalg.norm(gen_vec, axis=1) * np.linalg.norm(question_vec)
        cosine = np.dot(gen_vec, question_vec.T).reshape(-1) / norm
        return float(cosine.mean())
