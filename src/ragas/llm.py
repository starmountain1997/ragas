"""LLM access via agno structured output."""

import typing as t

from pydantic import BaseModel


class StructuredLLM:
    """Wraps an agno Model; generate() returns a parsed pydantic instance."""

    def __init__(self, model: t.Any):
        self.model = model

    async def generate(
        self,
        instruction: str,
        input_data: BaseModel,
        output_schema: type[BaseModel],
        examples: t.Sequence[tuple[BaseModel, BaseModel]] = (),
    ) -> BaseModel:
        # local import: tests inject fakes without agno installed deps
        from agno.agent import Agent

        prompt = instruction
        if examples:
            rendered = "\n\n".join(
                f"Example input:\n{i.model_dump_json()}\n"
                f"Example output:\n{o.model_dump_json()}"
                for i, o in examples
            )
            prompt += "\n\n" + rendered
        agent = Agent(model=self.model, output_schema=output_schema)
        response = await agent.arun(
            prompt + "\n\nInput:\n" + input_data.model_dump_json()
        )
        return response.content
