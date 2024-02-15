from typing import Tuple
from State import State
from enum import Enum
import copy
from typing import Set


class Player(Enum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2


class Hex(State):
    def __init__(self, board: list[list[Player]], player1_turn: bool):
        self.size = len(board)
        self.board = board
        self.player1_turn = player1_turn

    def get_legal_actions(self):
        actions: list[Tuple[int, int]] = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == Player.EMPTY:
                    actions.append((i, j))
        return actions

    def take_action(self, action: Tuple[int, int]):
        i, j = action
        new_board = copy.deepcopy(self.board)
        if self.player1_turn:
            new_board[i][j] = Player.PLAYER1
        else:
            new_board[i][j] = Player.PLAYER2
        return Hex(new_board, not self.player1_turn)

    def is_terminal(self):
        # Check if player 1 has won
        for i in range(self.size):
            if self.board[i][0] == Player.PLAYER1 and self.dfs(
                i, 0, Player.PLAYER1, set()
            ):
                return True

        # Check if player 2 has won
        for j in range(self.size):
            if self.board[0][j] == Player.PLAYER2 and self.dfs(
                0, j, Player.PLAYER2, set()
            ):
                return True

        return False

    def dfs(
        self, i: int, j: int, player: Player, visited: Set[Tuple[int, int]]
    ) -> bool:

        if (
            i < 0
            or i >= self.size
            or j < 0
            or j >= self.size
            or self.board[i][j] != player
        ):
            return False

        if (i, j) in visited:
            return False

        if player == Player.PLAYER1 and j == self.size - 1:
            return True

        if player == Player.PLAYER2 and i == self.size - 1:
            return True

        visited.add((i, j))

        return (
            self.dfs(i - 1, j, player, visited)
            or self.dfs(i + 1, j, player, visited)
            or self.dfs(i, j - 1, player, visited)
            or self.dfs(i, j + 1, player, visited)
            or self.dfs(i - 1, j + 1, player, visited)
            or self.dfs(i + 1, j - 1, player, visited)
        )

    def get_result(self):
        if not self.player1_turn:
            return 1
        else:
            return -1

    def __repr__(self):
        return "\n".join(
            [" ".join([str(cell.value) for cell in row]) for row in self.board]
        )

    def visualize(self):
        matrix(self.size, self.size, self.board)
        return


def matrix(n: int, m: int, li: list[list[Player]]):

    # Counter Variable
    ctr = 0
    while ctr < 2 * n - 1:
        print(" " * abs(n - ctr - 1), end="")
        lst: list[int] = []

        # Iterate [0, m]
        for i in range(m):

            # Iterate [0, n]
            for j in range(n):

                # Diagonal Elements
                # Condition
                if i + j == ctr:

                    # Appending the
                    # Diagonal Elements
                    lst.append(li[i][j].value)

        # Printing reversed Diagonal
        # Elements
        lst.reverse()
        print(*lst)
        ctr += 1
