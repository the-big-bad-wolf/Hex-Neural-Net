from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple


class State(ABC):
    def __init__(self, board):
        self.board: list
        self.player1_turn: bool

    @abstractmethod
    def get_legal_actions(self) -> list[Tuple[int, int]]:
        # Returning a list of legal actions from the current state
        pass

    @abstractmethod
    def get_state(self) -> list:
        # Returning the state of the game
        pass

    @abstractmethod
    def take_action(self, action: Tuple[int, int]) -> State:
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

    @abstractmethod
    def visualize(self) -> None:
        pass
