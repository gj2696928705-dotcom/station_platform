"""A02: explicit, frozen-input, four-slot simulation optimizer.

This package deliberately has no device or legacy plan-publishing side effects.
"""
from .contract import InputError, freeze, validate_input
from .solver import solve
from .verify import verify_plan, simulation_gate

__all__ = ["InputError", "freeze", "validate_input", "solve", "verify_plan", "simulation_gate"]
