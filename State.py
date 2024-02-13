from __future__ import annotations
from abc import ABC, abstractmethod


class State(ABC):
    def __init__(self):
        self.board = None
        self.player1_turn: bool

    @abstractmethod
    def get_legal_actions(self) -> list[int]:
        # Returning a list of legal actions from the current state
        pass

    @abstractmethod
    def take_action(self, action: int) -> State:
        # Returning the state that results from taking an action from the current state
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        # Checking if the current state is a terminal state
        pass

    @abstractmethod
    def get_result(self) -> float:
        pass

    def __repr__(self):
        return f"State({self.board}, {self.player1_turn})"
