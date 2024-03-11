from MonteCarloTreeSearch import MonteCarloTreeSearch, Node
from RLS import RLS
from HexDataset import HexDataset


class Controller:
    def __init__(
        self,
        MCTS: MonteCarloTreeSearch,
        RLS: RLS,
        RBUF_sample_size: int = 10,
        learning_rate: float = 0.01,
        training_epochs: int = 25,
        M: int = 10,
    ):
        self.MCTS = MCTS
        self.RLS = RLS
        self.RBUF = HexDataset([], [])
        self.RBUF_sample_size = RBUF_sample_size
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.M = M

    def run(self, nr_episodes: int):
        self.MCTS.ANET.save_model("./models/0episodes.pth")
        for i in range(1, nr_episodes + 1):
            print(f"Running episode {i}")
            self.run_episode()
            self.RLS.train(
                neural_net=self.MCTS.ANET,
                RBUF=self.RBUF,
                subset_size=self.RBUF_sample_size,
                epochs=self.training_epochs,
                learning_rate=self.learning_rate,
            )
            if i % self.M == 0:
                self.MCTS.ANET.save_model("./models/" + str(i) + "episodes.pth")
            self.MCTS.reset_root()

    def run_episode(self):
        while not self.MCTS.root.state.is_terminal():
            self.make_move()
            self.MCTS.root.state.visualize()

    def make_move(self):
        new_root = self.MCTS.search()
        self.RBUF.add_data(
            self.MCTS.root.state.get_state(), self.get_distibution(self.MCTS.root)
        )
        self.MCTS.root = new_root
        self.MCTS.root.parent = None

    def get_distibution(self, root: Node):

        D = [[0.0] * len(root.state.board) for _ in range(len(root.state.board))]

        sum_visits = 0
        for child in root.children:
            assert child.arcTo is not None
            sum_visits += child.visits
            D[child.arcTo[0]][child.arcTo[1]] = float(child.visits)

        for i in range(len(D)):
            for j in range(len(D[i])):
                if sum_visits != 0:
                    D[i][j] = D[i][j] / sum_visits
                else:
                    D[i][j] = 0.0
        return D
