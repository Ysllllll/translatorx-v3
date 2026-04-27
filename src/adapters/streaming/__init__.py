"""Streaming bus adapters.

Implementations of :class:`ports.message_bus.MessageBus`. The runtime
selects one based on :class:`ports.message_bus.BusConfig.type`.
"""

from .memory import InMemoryMessageBus

__all__ = ["InMemoryMessageBus"]
