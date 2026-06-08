import pytest
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentExtension,
)
from a2a.server.context import ServerCallContext
from a2a.server.agent_execution import RequestContext
from a2a_agent_certainty_extension.extension import (
    CertaintyTypes,
    CERTAINTY_FIELD_TYPE,
    CERTAINTY_FIELD_VALUE,
)
from a2a_agent_certainty_extension.extension import URI as EXTENSION_URI

from fixtures import certainty_extension, mock_agent_message


@pytest.mark.parametrize("supported", [True, False])
def test_is_supported(supported, certainty_extension):

    if supported:
        capabilities = AgentCapabilities(
            extensions=[AgentExtension(uri=EXTENSION_URI)],
        )
    else:
        capabilities = AgentCapabilities()
    mock_agent = AgentCard(
        capabilities=capabilities,
        default_input_modes=[],
        default_output_modes=[],
        description="Mock agent",
        name="mock_agent",
        documentation_url="www.mock_agent.com",
        version="0.0.1",
        skills=[],
    )
    assert certainty_extension.is_supported(mock_agent) == supported


@pytest.mark.parametrize(
    "metadata",
    [
        {
            CERTAINTY_FIELD_TYPE: CertaintyTypes.AVERAGE_TOKEN_PROBS,
            CERTAINTY_FIELD_VALUE: 0.0,
        },
        None,
    ],
    ids=["has_metadata", "no_metadata"],
)
def test_has_certainty(metadata, certainty_extension, mock_agent_message):

    agent_message = mock_agent_message(metadata)
    if metadata:
        assert certainty_extension.has_certainty(agent_message)
    else:
        assert not certainty_extension.has_certainty(agent_message)


@pytest.mark.parametrize("certainty_type", list(CertaintyTypes))
@pytest.mark.parametrize("certainty_value", [0.0, 0.2, 0.9, 1.0])
def test_get_certainty(
    certainty_type, certainty_value, certainty_extension, mock_agent_message
):

    agent_message = mock_agent_message(
        {CERTAINTY_FIELD_TYPE: certainty_type, CERTAINTY_FIELD_VALUE: certainty_value}
    )

    certainty_data = certainty_extension.get_certainty(agent_message)

    assert certainty_type in certainty_data
    assert certainty_data[certainty_type] == certainty_value


@pytest.mark.parametrize("certainty_type", list(CertaintyTypes))
@pytest.mark.parametrize("certainty_value", [-1.0, -1e-10, 1.01, 100.0])
def test_get_certainty_out_of_bounds(
    certainty_type, certainty_value, certainty_extension, mock_agent_message
):

    agent_message = mock_agent_message(
        {CERTAINTY_FIELD_TYPE: certainty_type, CERTAINTY_FIELD_VALUE: certainty_value}
    )

    with pytest.raises(ValueError, match="Invalid certainty value"):
        certainty_extension.get_certainty(agent_message)


def test_get_certainty_no_certainty(certainty_extension, mock_agent_message):

    agent_message = mock_agent_message()
    assert certainty_extension.get_certainty(agent_message) is None


@pytest.mark.parametrize("certainty_type", list(CertaintyTypes))
@pytest.mark.parametrize("certainty_value", [0.0, 0.2, 0.9, 1.0])
def test_add_certainty_valid(
    certainty_type, certainty_value, mock_agent_message, certainty_extension
):

    agent_message = mock_agent_message()

    certainty_extension.add_certainty(agent_message, certainty_type, certainty_value)

    assert CERTAINTY_FIELD_TYPE in agent_message.metadata
    assert CERTAINTY_FIELD_VALUE in agent_message.metadata
    assert agent_message.metadata[CERTAINTY_FIELD_TYPE] == certainty_type
    assert agent_message.metadata[CERTAINTY_FIELD_VALUE] == certainty_value
    assert EXTENSION_URI in agent_message.extensions


@pytest.mark.parametrize("certainty_type", list(CertaintyTypes))
@pytest.mark.parametrize("certainty_value", [0.0, 0.2, 0.9, 1.0])
def test_add_certainty_already_present(
    certainty_type, certainty_value, mock_agent_message, certainty_extension
):

    agent_message = mock_agent_message(
        {
            CERTAINTY_FIELD_TYPE: CertaintyTypes.AVERAGE_TOKEN_PROBS,
            CERTAINTY_FIELD_VALUE: 0.5,
        },
        [EXTENSION_URI],
    )

    certainty_extension.add_certainty(agent_message, certainty_type, certainty_value)

    assert CERTAINTY_FIELD_TYPE in agent_message.metadata
    assert CERTAINTY_FIELD_VALUE in agent_message.metadata
    assert (
        agent_message.metadata[CERTAINTY_FIELD_TYPE]
        == CertaintyTypes.AVERAGE_TOKEN_PROBS
    )
    assert agent_message.metadata[CERTAINTY_FIELD_VALUE] == 0.5
    assert EXTENSION_URI in agent_message.extensions


@pytest.mark.parametrize("certainty_type", list(CertaintyTypes))
@pytest.mark.parametrize("certainty_value", [-2.0, -1e-10, 1.01, 100.0])
def test_add_certainty_invalid(
    certainty_type, certainty_value, mock_agent_message, certainty_extension
):

    agent_message = mock_agent_message()

    with pytest.raises(ValueError, match="Certainty value must be in range of"):
        certainty_extension.add_certainty(
            agent_message, certainty_type, certainty_value
        )


def test_agent_extension(certainty_extension):

    agent_extension = certainty_extension.agent_extension()
    assert agent_extension.uri == EXTENSION_URI


@pytest.mark.parametrize("requested", [True, False])
def test_activate(requested, certainty_extension):

    if requested:
        context = RequestContext(
            call_context=ServerCallContext(requested_extensions={EXTENSION_URI})
        )
        assert EXTENSION_URI in context.requested_extensions
    else:
        context = RequestContext(call_context=ServerCallContext())
    assert requested == certainty_extension.is_requested(context)
