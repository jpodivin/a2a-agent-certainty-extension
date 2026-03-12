# OpenAI API adapter

This example demonstrates how to integrate the `a2a-agent-certainty-extension` with a standard 
OpenAI-compatible API to generate content alongside confidence scores.
It features a trivia agent that can report its certainty using different methods.

## Overview

The `trivia.py` script implements a custom `OpenAITriviaAdapter` that inherits from `LLMBackendAdapter`.
This adapter interacts with an LLM (configured via environment variables) to fetch trivia facts
and calculate a certainty score based on the requested method.

## Certainty Methods Supported

The adapter supports two primary ways of determining confidence:

* **Self-Reported**: The agent is prompted to return a JSON object containing
    both the trivia fact and a `certainty_score` between 0.0 and 1.0.
* **Average Token Probabilities**: The adapter requests `logprobs` from the OpenAI API
    and calculates the mathematical average of the token probabilities to derive a confidence score.

## Configuration

The script uses the following environment variables for configuration:

* `MODEL_NAME`: The name of the model to use (defaults to a specific `granite` GGUF path).
* `LLM_SERVER_URL`: The base URL for the OpenAI-compatible server (defaults to `http://127.0.0.1:8080`).
* `OPENAI_API_KEY`: The API key required for authentication.

## Usage

You can run the example directly from the command line. By default, it uses the `SELF_REPORTED` certainty type.

### Basic Execution
```bash
python trivia.py
```

### Specifying Certainty Type
You can choose the method used to calculate confidence using the `--certainty-type` flag:

```bash
# To use average token probabilities
python trivia.py --certainty-type average_token_probs

# To use self-reported JSON scores
python trivia.py --certainty-type self_reported
```
