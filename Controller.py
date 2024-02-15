from Nim import Nim
from MonteCarloTreeSearch import Node, MonteCarloTreeSearch
import random

# random.seed(123)

Nim = Nim(10, 5, True)
MCTS = MonteCarloTreeSearch(1, Node(Nim, None))

best_node = MCTS.search(500)
print(best_node.state)
