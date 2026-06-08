#! /usr/bin/env python

import asyncio
import argparse
import os
import json
import math
from uuid import uuid4
from openai import AsyncOpenAI

from a2a.types import Message, Role, Part
from a2a.server.agent_execution import RequestContext
from a2a.server.events.event_queue import EventQueue

from a2a_agent_certainty_extension.executors import (
    LLMBackendAdapter,
    CertaintyAgentExecutor,
)
from a2a_agent_certainty_extension.extension import CertaintyTypes, CertaintyExtension

MODEL_NAME = os.environ.get(
    "MODEL_NAME", "repos/logdetective/models/granite-4.0-h-tiny-Q8_0.gguf"
)
LLM_SERVER_URL = os.environ.get("LLM_SERVER_URL", "http://127.0.0.1:8080")


class OpenAITriviaAdapter(LLMBackendAdapter):
    """
    Example implementation of LLMBackendAdapter that focuses on setup.
    It uses the OpenAI API to generate a trivia fact and populate its system prompt.
    """

    def __init__(self, certainty_type: CertaintyTypes = CertaintyTypes.SELF_REPORTED):
        # Initialize the OpenAI client using the official library
        self.client = AsyncOpenAI(
            base_url=LLM_SERVER_URL, api_key=os.environ.get("OPENAI_API_KEY", "")
        )
        self.certainty_type = certainty_type

        if certainty_type == CertaintyTypes.SELF_REPORTED:
            self.system_prompt = (
                "You are a helpful trivia agent. Your task is to provide a single, "
                "interesting random trivia fact. You must respond strictly in JSON format "
                "with the following keys:\n"
                "- 'fact': The trivia fact string.\n"
                "- 'certainty_score': A float between 0.0 and 1.0 representing your confidence in the fact."
            )
        else:
            self.system_prompt = (
                "You are a helpful trivia agent. Your task is to provide a single, "
                "interesting random trivia fact."
            )

    async def generate_with_certainty(
        self, message: Message, *args, **kwargs
    ) -> tuple[str, CertaintyTypes, float]:
        """
        Calls OpenAI to generate a trivia fact along with its certainty score.
        """
        c_score = 0.0
        # Prepare messages combining the system prompt and user input
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": getattr(message, "content", str(message))},
        ]

        create_kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
        }

        if self.certainty_type == CertaintyTypes.AVERAGE_TOKEN_PROBS:
            create_kwargs["logprobs"] = True

        if self.certainty_type == CertaintyTypes.SELF_REPORTED:
            create_kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**create_kwargs)
        if not response.choices[0].message.content:
            raise ValueError("No response from LLM")
        # Parse response
        raw_content = response.choices[0].message.content.strip()

        if self.certainty_type == CertaintyTypes.SELF_REPORTED:
            content = json.loads(raw_content)
            fact = content.get("fact", "No fact generated.")
            c_score = float(content.get("certainty_score", 0.0))
        else:
            fact = raw_content

        if self.certainty_type == CertaintyTypes.AVERAGE_TOKEN_PROBS:
            logprobs_content = response.choices[0].logprobs.content
            if logprobs_content:
                probs = [math.exp(token.logprob) for token in logprobs_content]
                c_score = sum(probs) / len(probs)
            else:
                c_score = 0.0

        return fact, self.certainty_type, c_score


async def main(certainty_type: CertaintyTypes):

    # Instantiate the adapter
    adapter = OpenAITriviaAdapter(certainty_type)

    # Create a message to trigger the agent
    print("Generating a trivia fact...\n")

    event_queue = EventQueue()

    user_msg = Message(
        role=Role.ROLE_USER,
        parts=[Part(text="Tell me something interesting!")],
        message_id=str(uuid4()),
    )

    context = RequestContext(request=MessageSendParams(message=user_msg))
    executor = CertaintyAgentExecutor(
        llm_backend_adapter=adapter, certainty_extension=CertaintyExtension()
    )
    # Execute generate_with_certainty

    await executor.execute(context=context, event_queue=event_queue)

    event = await event_queue.queue.get()
    if not isinstance(event, Message):
        raise TypeError("Retrieved event is not a Message object!")

    reported_certainty = None
    response = None
    reported_certainty_type = None

    for part in event.parts:
        if isinstance(part.root, TextPart):
            response = part.root.text
        if isinstance(part.root, DataPart):
            certainty_data = part.root.data

            reported_certainty = certainty_data["certainty"]
            reported_certainty_type = CertaintyTypes(certainty_data["certainty_type"])
    if not (reported_certainty_type and reported_certainty and response):
        raise RuntimeError(f"Invalid response from executor {event}")

    print("Trivia Fact:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    print(
        f"Confidence: {reported_certainty_type.name} ({reported_certainty * 100:.2f}%)\n"
    )


def cli():
    """
    CLI entrypoint for `trivia` generator.
    """
    parser = argparse.ArgumentParser(
        description="Run the Trivia OpenAI Adapter Example"
    )
    parser.add_argument(
        "--certainty-type", default=CertaintyTypes.SELF_REPORTED, choices=CertaintyTypes
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.certainty_type))
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    cli()
