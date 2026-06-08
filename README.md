# Certainty Extension

## Overview

This extension adds fields containing measure of a model certainty to `Message` and `Artifact` objects.

## Extension URI

The URI of this extension is `https://github.com/jpodivin/a2a-agent-certainty-extension/docs/v1`.

This is the only URI accepted for this extension.

## Certainty Value Format

Certainty value MUST be a float, in range of <0.0,1.0> inclusive, regardless of the Certainty Type.

## Certainty Type Format

Certainty type MUST be one of allowed values, either `AVERAGE_TOKEN_PROBS` or `SELF_REPORTED`.

No other values are allowed.

## Message/Artifact Metadata Fields

Certainty value MUST be stored in the metadata for a Message or Artifact, under a
field with the key `github.com/jpodivin/a2a-agent-certainty-extension/docs/v1/certainty_value`.

The value of certainty MUST be a float adhering to the [Certainty Value Format](#certainty-value-format).

Certainty type MUST be stored in the metadata for a Message or Artifact, under a
field with the key `github.com/jpodivin/a2a-agent-certainty-extension/docs/v1/certainty_type`.

The value of certainty type MUST be a string adhering to the [Certainty Type Format](#certainty-type-format).

When this extension is applied to a Message or Artifact, the Extension URI MUST also be appended
to the extensions array of that object (for clients utilizing A2A v1.0 or higher).

## Certainty Generation

Certainty value MUST be generated, using a method corresponding to one of allowed Certainty Types,
either `AVERAGE_TOKEN_PROBS` or `SELF_REPORTED`.


## Extension Activation

Clients indicate their desire to receive certainty values and types
on messages by specifying the [Extension URI](#extension-uri) via the
transport-defined extension activation mechanism.
For JSON-RPC and HTTP transports, this is indicated via the `A2A-Extensions`
HTTP header. For gRPC, this is indicated via the `A2A-Extensions` metadata value.