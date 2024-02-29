from MonteCarloTreeSearch import MonteCarloTreeSearch, Node
from NeuralNet import NeuralNet


class Controller:
    def __init__(
        self,
        MCTS: MonteCarloTreeSearch,
        ANET: NeuralNet,
    ):
        self.MCTS = MCTS
        self.ANET = ANET
        self.RBUF = []

    def run(self, nr_episodes: int):
        for _ in range(nr_episodes):
            self.run_episode()
            self.ANET.train(self.RBUF)

    def run_episode(self):
        while not self.MCTS.root.state.is_terminal():
            self.MCTS.root.state.visualize()
            self.make_move()
        self.MCTS.root.state.visualize()

    def make_move(self):
        new_root = self.MCTS.search()
        self.RBUF.append(
            (self.MCTS.root.state.get_state(), self.get_distibution(self.MCTS.root))
        )
        self.MCTS.root = new_root
        self.MCTS.root.parent = None

    def get_distibution(self, root: Node):

        D = [[float(0)] * len(root.state.board) for _ in range(len(root.state.board))]

        sum_visits = 0
        for child in root.children:
            assert child.arcTo is not None
            sum_visits += child.visits
            D[child.arcTo[0]][child.arcTo[1]] = child.visits

        for i in range(len(D)):
            for j in range(len(D[i])):
                if sum_visits != 0:
                    D[i][j] = D[i][j] / sum_visits
                else:
                    D[i][j] = 0

        return D
