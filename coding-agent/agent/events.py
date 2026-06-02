"""Event sink boundary so the agent turn loop drives both the terminal REPL and
the AG-UI web server without duplicating the loop logic.

run_turn (agent/agent.py) keeps all its existing Rich console.print calls — those
remain the CLI's visible output and go to server stdout under the web server. On top
of that it emits three *structured* signals through an EventSink:

  - tool_call(name, arguments) — a tool is about to run
  - flow_built(flow_id)        — a flow was created/updated/built (drives the canvas)
  - final(text)                — the assistant's end-of-turn answer

ConsoleSink is a no-op: the CLI already renders everything via console.print, so the
structured signals add nothing there and CLI behaviour stays identical. The web sink
(AGUISink, in server/) maps these signals to AG-UI protocol events.
"""
from typing import Protocol


class EventSink(Protocol):
    def tool_call(self, name: str, arguments: dict) -> None: ...
    def flow_built(self, flow_id: str | None) -> None: ...
    def final(self, text: str | None) -> None: ...


class ConsoleSink:
    """No-op sink for the terminal REPL — existing console.print output is unchanged."""

    def tool_call(self, name: str, arguments: dict) -> None:
        pass

    def flow_built(self, flow_id: str | None) -> None:
        pass

    def final(self, text: str | None) -> None:
        pass
