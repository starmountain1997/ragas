import numpy as np
from pydantic import BaseModel, Field

from ragas.metrics.base import Metric
from ragas.schema import SingleTurnSample


class StatementGeneratorInput(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")


class StatementGeneratorOutput(BaseModel):
    statements: list[str] = Field(description="The generated statements")


STATEMENT_GENERATOR_INSTRUCTION = (
    "Given a question and an answer, analyze the complexity of each sentence in the "
    "answer. Break down each sentence into one or more fully understandable "
    "statements. Ensure that no pronouns are used in any statement. Format the "
    "outputs in JSON."
)
STATEMENT_GENERATOR_EXAMPLES = [
    (
        StatementGeneratorInput(
            question="Who was Albert Einstein and what is he best known for?",
            answer="He was a German-born theoretical physicist, widely acknowledged "
            "to be one of the greatest and most influential physicists of all time. "
            "He was best known for developing the theory of relativity, he also made "
            "important contributions to the development of the theory of quantum "
            "mechanics.",
        ),
        StatementGeneratorOutput(
            statements=[
                "Albert Einstein was a German-born theoretical physicist.",
                (
                    "Albert Einstein is recognized as one of the greatest and most "
                    "influential physicists of all time."
                ),
                "Albert Einstein was best known for developing the theory of relativity.",
                (
                    "Albert Einstein also made important contributions to the development "
                    "of the theory of quantum mechanics."
                ),
            ]
        ),
    )
]


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: list[str] = Field(..., description="The statements to judge")


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class NLIStatementOutput(BaseModel):
    statements: list[StatementFaithfulnessAnswer]


NLI_INSTRUCTION = (
    "Your task is to judge the faithfulness of a series of statements based on a "
    "given context. For each statement you must return verdict as 1 if the statement "
    "can be directly inferred based on the context or 0 if the statement can not be "
    "directly inferred based on the context."
)
NLI_EXAMPLES = [
    (
        NLIStatementInput(
            context="""John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
            statements=[
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
        ),
        NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(
                    statement="John is majoring in Biology.",
                    reason="John's major is explicitly mentioned as Computer Science. "
                    "There is no information suggesting he is majoring in Biology.",
                    verdict=0,
                ),
                StatementFaithfulnessAnswer(
                    statement="John is taking a course on Artificial Intelligence.",
                    reason="The context mentions the courses John is currently enrolled "
                    "in, and Artificial Intelligence is not mentioned. Therefore, it "
                    "cannot be deduced that John is taking a course on AI.",
                    verdict=0,
                ),
                StatementFaithfulnessAnswer(
                    statement="John is a dedicated student.",
                    reason="The context states that he spends a significant amount of "
                    "time studying and completing assignments. Additionally, it "
                    "mentions that he often stays late in the library to work on his "
                    "projects, which implies dedication.",
                    verdict=1,
                ),
                StatementFaithfulnessAnswer(
                    statement="John has a part-time job.",
                    reason="There is no information given in the context about John "
                    "having a part-time job.",
                    verdict=0,
                ),
            ]
        ),
    ),
    (
        NLIStatementInput(
            context="Photosynthesis is a process used by plants, algae, and certain "
            "bacteria to convert light energy into chemical energy.",
            statements=["Albert Einstein was a genius."],
        ),
        NLIStatementOutput(
            statements=[
                StatementFaithfulnessAnswer(
                    statement="Albert Einstein was a genius.",
                    reason="The context and statement are unrelated",
                    verdict=0,
                )
            ]
        ),
    ),
]


class Faithfulness(Metric):
    def __init__(self, llm=None, embeddings=None, name: str = "faithfulness"):
        super().__init__(
            name=name,
            required_columns={"user_input", "response", "retrieved_contexts"},
            llm=llm,
            embeddings=embeddings,
        )

    async def single_turn_ascore(self, sample: SingleTurnSample) -> float:
        assert self.llm is not None, "LLM is not set"

        statements = await self.llm.generate(
            STATEMENT_GENERATOR_INSTRUCTION,
            StatementGeneratorInput(
                question=sample.user_input, answer=sample.response or ""
            ),
            StatementGeneratorOutput,
            STATEMENT_GENERATOR_EXAMPLES,
        )
        if not statements.statements:
            return np.nan

        contexts_str = "\n".join(sample.retrieved_contexts or [])
        verdicts = await self.llm.generate(
            NLI_INSTRUCTION,
            NLIStatementInput(context=contexts_str, statements=statements.statements),
            NLIStatementOutput,
            NLI_EXAMPLES,
        )
        faithful = sum(1 for a in verdicts.statements if a.verdict)
        return faithful / len(verdicts.statements)
