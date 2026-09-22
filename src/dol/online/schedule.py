"""DOLのAwake/Hibernate周期を明示的な状態機械で表す。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseTransition:
    previous: str
    current: str

    @property
    def changed(self) -> bool:
        return self.previous != self.current

    @property
    def entered_hibernation(self) -> bool:
        return self.previous == "awake" and self.current == "hibernate"

    @property
    def entered_awake(self) -> bool:
        return self.previous == "hibernate" and self.current == "awake"


class AwakeHibernateSchedule:
    """公開実装のcountとmodulo判定をそのまま明示化した状態機械。

    公開実装は現在stepの予測後、countを増やす前に位相を
    切り替える。そのため最初のAwakeはstep 0から
    ``awake_steps`` までを含む。
    """

    def __init__(self, awake_steps: int, hibernate_steps: int) -> None:
        if awake_steps <= 0 or hibernate_steps <= 0:
            raise ValueError("awake_steps and hibernate_steps must be positive")
        self.awake_steps = awake_steps
        self.hibernate_steps = hibernate_steps
        self.phase = "awake"
        self.step = 0
        self.steps_in_phase = 0

    @property
    def is_awake(self) -> bool:
        return self.phase == "awake"

    def advance(self) -> PhaseTransition:
        """現在stepの処理後に公開実装と同じ条件で位相を変更する。"""

        previous = self.phase
        cycle_steps = self.awake_steps + self.hibernate_steps
        if self.step != 0 and self.step % self.awake_steps == 0 and self.is_awake:
            self.phase = "hibernate"
            self.steps_in_phase = 0
        elif self.step % cycle_steps == 0 and not self.is_awake:
            self.phase = "awake"
            self.steps_in_phase = 0
        else:
            self.steps_in_phase += 1
        self.step += 1
        return PhaseTransition(previous=previous, current=self.phase)

    def reset(self) -> None:
        self.phase = "awake"
        self.step = 0
        self.steps_in_phase = 0
