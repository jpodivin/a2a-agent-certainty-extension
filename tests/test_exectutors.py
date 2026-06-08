import pytest

from a2a.server.agent_execution import RequestContext
from a2a.types import Message, Role, SendMessageRequest
from a2a.server.context import ServerCallContext


from a2a_agent_certainty_extension.extension import CertaintyTypes
from a2a_agent_certainty_extension.extension import URI as EXTENSION_URI

from fixtures import (
    mock_agent_message,
    mock_event_queue,
    mock_llm_adapter,
    executor,
    certainty_extension,
)


@pytest.mark.asyncio
async def test_execute_with_certainty_activated(
    executor,
    mock_llm_adapter,
    mock_event_queue,
    mock_agent_message,
):
    message_params = SendMessageRequest(message=mock_agent_message())
    # Setup: Extension is requested
    context = RequestContext(
        context_id="test-session",
        call_context=ServerCallContext(requested_extensions={EXTENSION_URI}),
        request=message_params,
    )

    # Mock LLM response
    mock_llm_adapter.generate_with_certainty.return_value = (
        "This is the agent response.",
        CertaintyTypes.AVERAGE_TOKEN_PROBS,
        0.95,
    )

    # Execution
    await executor.execute(context, mock_event_queue)

    # Verification: Check that the LLM was called correctly
    mock_llm_adapter.generate_with_certainty.assert_called_once_with(context.message)

    # Verification: Check the enqueued message
    mock_event_queue.enqueue_event.assert_called_once()
    sent_message = mock_event_queue.enqueue_event.call_args[0][0]

    assert isinstance(sent_message, Message)
    assert sent_message.role == Role.ROLE_AGENT
    assert sent_message.parts[0].text == "This is the agent response."

    # Check that certainty metadata was added because it was activated
    from a2a_agent_certainty_extension.extension import (
        CERTAINTY_FIELD_TYPE,
        CERTAINTY_FIELD_VALUE,
    )

    assert sent_message.extensions
    assert sent_message.metadata
    assert (
        sent_message.metadata[CERTAINTY_FIELD_TYPE]
        == CertaintyTypes.AVERAGE_TOKEN_PROBS
    )
    assert sent_message.metadata[CERTAINTY_FIELD_VALUE] == 0.95
    assert EXTENSION_URI in sent_message.extensions


@pytest.mark.asyncio
async def test_execute_without_certainty_activated(
    executor,
    mock_llm_adapter,
    mock_event_queue,
    mock_agent_message,
):
    message_params = SendMessageRequest(message=mock_agent_message())
    # Setup: Extension is NOT requested
    context = RequestContext(
        context_id="test-session",
        call_context=ServerCallContext(requested_extensions=set()),
        request=message_params,
    )

    mock_llm_adapter.generate_with_certainty.return_value = (
        "Response without certainty metadata.",
        CertaintyTypes.SELF_REPORTED,
        0.8,
    )

    # Execution
    await executor.execute(context, mock_event_queue)

    # Verification
    sent_message = mock_event_queue.enqueue_event.call_args[0][0]

    # Metadata should be empty because extension wasn't activated in context
    assert sent_message.metadata == {}
    assert (
        sent_message.extensions is None or EXTENSION_URI not in sent_message.extensions
    )


@pytest.mark.asyncio
async def test_execute_no_message_in_context(
    executor, mock_event_queue, mock_llm_adapter
):
    # Setup: Context has no message (e.g., initial connection or malformed request)
    context = RequestContext(call_context=ServerCallContext(requested_extensions=set()), context_id="empty-context")

    # Execution
    await executor.execute(context, mock_event_queue)

    # Verification: LLM shouldn't be called, nothing enqueued
    mock_llm_adapter.generate_with_certainty.assert_not_called()
    mock_event_queue.enqueue_event.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_with_task_id(executor, mock_event_queue):
    # Setup: Context with a task ID to cancel
    context = RequestContext(call_context=ServerCallContext(requested_extensions=set()), context_id="test-session", task_id="task-123")

    # Execution
    await executor.cancel(context, mock_event_queue)

    # Verification: A message should be enqueued (likely a Task rejection/status update)
    # The TaskUpdater handles the internal details, but we expect interaction with the queue.
    mock_event_queue.enqueue_event.assert_called_once()
    event = mock_event_queue.enqueue_event.call_args[0][0]

    # A2A TaskUpdater usually sends a Message with metadata indicating rejection/status
    assert event.context_id == "test-session"
