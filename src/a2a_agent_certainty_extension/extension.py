from typing import Optional, Dict
from enum import StrEnum

from a2a.server.agent_execution import RequestContext
from a2a.types import (
    AgentCard,
    AgentExtension,
    Artifact,
    Message,
)
from a2a.extensions.common import find_extension_by_uri

_CORE_PATH = "github.com/jpodivin/a2a-agent-certainty-extension/docs/v1"
URI = f"https://{_CORE_PATH}"
CERTAINTY_FIELD_VALUE = f"{_CORE_PATH}/certainty_value"
CERTAINTY_FIELD_TYPE = f"{_CORE_PATH}/certainty_type"


class CertaintyTypes(StrEnum):
    """Allowed types of certainty."""

    SELF_REPORTED = "SELF_REPORTED"
    AVERAGE_TOKEN_PROBS = "AVERAGE_TOKEN_PROBS"


class CertaintyExtension:
    """A2A extension providing agents with means to express a measure of
    certainty about their responses."""

    def is_supported(self, agent_card: AgentCard) -> bool:
        """Check that the extension is supported by the AgentCard"""

        return find_extension_by_uri(agent_card, URI) is not None

    def has_certainty(self, struct: Message | Artifact) -> bool:
        """Check that structure has certainty field in metadata"""
        if struct.metadata:
            return (
                CERTAINTY_FIELD_VALUE in struct.metadata
                and CERTAINTY_FIELD_TYPE in struct.metadata
            )
        return False

    def get_certainty(self, struct: Message | Artifact) -> Optional[Dict[str, float]]:
        """Return certainty if it is set in the structure"""
        if struct.metadata and self.has_certainty(struct):
            certainty_type = struct.metadata[CERTAINTY_FIELD_TYPE]
            certainty_value = struct.metadata[CERTAINTY_FIELD_VALUE]

            if isinstance(certainty_type, str) and isinstance(certainty_value, float):
                if 0 <= certainty_value <= 1:
                    return {certainty_type: certainty_value}
                raise ValueError(
                    f"Invalid certainty value: {certainty_value}, received."
                )
        return None

    def agent_extension(self) -> AgentExtension:
        """Get AgentExtension object"""
        return AgentExtension(
            uri=URI, description="Adds certainty to messages and artifacts"
        )

    def is_requested(self, context: RequestContext) -> bool:
        """Return true if client requested this extension for the call.
        """
        return URI in context.requested_extensions

    def add_certainty(
        self,
        struct: Message | Artifact,
        certainty_type: CertaintyTypes,
        certainty_value: float,
    ) -> None:
        """Add certainty if it isn't already set, add certainty extension URI to the extensions array"""

        if not 0 <= certainty_value <= 1:
            raise ValueError(
                f"Certainty value must be in range of <0.0,1.0> inclusive, certainty value: {certainty_value}."
            )

        # Do not change certainty if it is already set
        if self.has_certainty(struct):
            return

        struct.metadata[CERTAINTY_FIELD_VALUE] = certainty_value
        struct.metadata[CERTAINTY_FIELD_TYPE] = certainty_type

        struct.extensions.append(URI)

    def add_if_requested(
        self,
        struct: Message | Artifact,
        context: RequestContext,
        certainty_type: CertaintyTypes,
        certainty_value: float,
    ) -> None:
        """Add certainty to struct only if it is requested"""
        if self.is_requested(context):
            self.add_certainty(
                struct, certainty_type=certainty_type, certainty_value=certainty_value
            )
