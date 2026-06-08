import pytest

from a2a.types import (
    Role,
    Message,
)

from a2a_agent_certainty_extension.extension import (
    CertaintyExtension,
)

from unittest.mock import AsyncMock

from a2a.server.events.event_queue import EventQueue

from a2a_agent_certainty_extension.executors import (
    CertaintyAgentExecutor,
    LLMBackendAdapter,
)
from a2a_agent_certainty_extension.extension import CertaintyExtension


@pytest.fixture
def mock_agent_message():

    def message_builder(metadata=None, extensions=None):
        return Message(
            context_id="123",
            message_id="abc",
            parts=[],
            role=Role.ROLE_AGENT,
            metadata=metadata,
            extensions=extensions,
        )

    return message_builder


@pytest.fixture
def mock_llm_adapter():
    return AsyncMock(spec=LLMBackendAdapter)


@pytest.fixture
def certainty_extension():
    return CertaintyExtension()


@pytest.fixture
def executor(mock_llm_adapter, certainty_extension):
    return CertaintyAgentExecutor(
        llm_backend_adapter=mock_llm_adapter,
        certainty_extension=certainty_extension,
    )


@pytest.fixture
def mock_event_queue():
    return AsyncMock(spec=EventQueue)
