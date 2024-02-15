from Hex import Hex, Player
from MonteCarloTreeSearch import Node, MonteCarloTreeSearch

# import random

# random.seed(123)

board: list[list[Player]] = []

for i in range(5):
    row: list[Player] = []
    for j in range(5):
        row.append(Player.EMPTY)
    board.append(row)


Hex = Hex(board, True)
MCTS = MonteCarloTreeSearch(1, Node(Hex, None))

best_node = MCTS.search(500)
best_node.state.visualize()
