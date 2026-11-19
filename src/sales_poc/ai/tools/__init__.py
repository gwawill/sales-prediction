"""Read-only tools the agent can call (task T12, FR-27 and FR-29)."""


class ToolError(ValueError):
    """A tool refused a request or failed. The message is meant to be shown to the model."""
