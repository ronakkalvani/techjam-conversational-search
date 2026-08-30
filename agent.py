"""Organizer-facing entrypoint for the EntropyShop submission.

The implementation remains in ``starter.agent`` so the development harness
and package layout stay unchanged.  This module exposes the single ``Agent``
class expected by the organizer's submission loader.
"""

from starter.agent import Agent

__all__ = ["Agent"]
