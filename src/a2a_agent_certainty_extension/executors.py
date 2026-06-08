from typing import Protocol
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Message, Part, Role

from .extension import CertaintyExtension, CertaintyTypes


class LLMBackendAdapter(Protocol):
    async def generate_with_certainty(
        self, message: Message, *args, **kwargs
    ) -> tuple[str, CertaintyTypes, float]:
        """Call LLM backend and generate response with some measure of certainty.
        Return tuple of response, string identifying the type of certainty generated and a certainty value.
        The value of certainty must be a float, with higher values indicating higher certainty."""
        ...


class CertaintyAgentExecutor(AgentExecutor):
    """Custom executor, adding information necessary for generating certainty
    to structs produced by the agent."""

    def __init__(
        self,
        llm_backend_adapter: LLMBackendAdapter,
        certainty_extension: CertaintyExtension,
    ) -> None:
        self.llm_backend_adapter = llm_backend_adapter
        self.certainty_extension = certainty_extension

        super().__init__()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            return
        user_message = context.message
        (
            llm_response,
            certainty_type,
            certainty,
        ) = await self.llm_backend_adapter.generate_with_certainty(user_message)

        response = Message(
            context_id=context.context_id,
            message_id=str(uuid4()),
            role=Role.ROLE_AGENT,
            parts=[
                Part(text=llm_response),
            ],
            metadata={},
        )

        self.certainty_extension.add_if_requested(
            response,
            context=context,
            certainty_type=certainty_type,
            certainty_value=certainty,
        )

        await event_queue.enqueue_event(response)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Reject the referenced task if one is supplied."""
        if context.task_id:
            updater = TaskUpdater(
                event_queue,
                task_id=context.task_id,
                context_id=context.context_id or str(uuid4()),
            )
            await updater.reject()
