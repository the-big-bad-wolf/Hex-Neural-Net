from State import State


class Nim(State):
    def __init__(
        self,
        initial_stones: int,
        max_stones_removed: int,
        player1_turn: bool,
    ):
        self.stones = initial_stones
        self.max_stones_removed = max_stones_removed
        self.player1_turn = player1_turn

    def get_legal_actions(self):
        # Returning a list of legal actions from the current state
        actions: list[int] = []
        for i in range(1, min(self.max_stones_removed, self.stones) + 1):
            actions.append(i)
        return actions

    def take_action(self, action: int):
        # Returning the state that results from taking an action from the current state
        return Nim(
            self.stones - action,
            self.max_stones_removed,
            not self.player1_turn,
        )

    def is_terminal(self):
        # Checking if the current state is a terminal state
        if self.stones == 0:
            return True
        return False

    def get_result(self):
        return 1 if not self.player1_turn else -1

    def __repr__(self):
        return f"State({self.stones}, {self.player1_turn})"
