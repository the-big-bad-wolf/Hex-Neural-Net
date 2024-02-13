from Nim import Nim
from MonteCarloTreeSearch import Node, MonteCarloTreeSearch

Nim = Nim(10, 5, True)
MCTS = MonteCarloTreeSearch(200, Node(Nim, None))

best_node = MCTS.search(500)
print(best_node.state)
