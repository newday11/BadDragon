"""Explicit task-step state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepState(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    DECIDE = "DECIDE"
    DONE = "DONE"
    FAIL = "FAIL"


ALLOWED_TRANSITIONS: dict[StepState, set[StepState]] = {
    StepState.PLAN: {StepState.ACT, StepState.FAIL},
    StepState.ACT: {StepState.OBSERVE, StepState.FAIL},
    StepState.OBSERVE: {StepState.DECIDE, StepState.FAIL},
    StepState.DECIDE: {StepState.ACT, StepState.DONE, StepState.FAIL},
    StepState.DONE: set(),
    StepState.FAIL: set(),
}


@dataclass
class StepStateMachine:
    step_id: int
    state: StepState = StepState.PLAN
    history: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, to_state: StepState, reason: str) -> tuple[bool, str]:
        allowed = ALLOWED_TRANSITIONS.get(self.state, set())
        ok = to_state in allowed
        message = reason
        if not ok:
            message = f"illegal_transition:{self.state.value}->{to_state.value}:{reason}"
        self.history.append(
            {
                "from": self.state.value,
                "to": to_state.value,
                "ok": ok,
                "reason": message,
            }
        )
        if ok:
            self.state = to_state
        return ok, message
