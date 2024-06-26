from MonteCarloTreeSearch import MonteCarloTreeSearch, Node
from RLS import RLS
import csv


class Controller:
    def __init__(
        self,
        MCTS: MonteCarloTreeSearch,
        RLS: RLS,
        RBUF_sample_size: int = 10,
        learning_rate: float = 0.01,
        training_epochs: int = 25,
        M: int = 10,
        visualize: bool = False,
    ):
        self.MCTS = MCTS
        self.RLS = RLS
        self.RBUF: list[tuple[list[float], list[list[float]]]] = []
        self.RBUF_sample_size = RBUF_sample_size
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.M = M
        self.visualize = visualize

        def load_training_data(file_path):
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                features: list[list[float]] = []
                targets: list[list[list[float]]] = []
                for row in reader:
                    feature: list[float] = []
                    target: list[list[float]] = []
                    for square in row[0 : len(MCTS.root.state.get_state())]:
                        feature.append(float(square))
                    features.append(feature)
                    for target_row in row[len(MCTS.root.state.get_state()) :]:
                        target_row_array = []
                        for probability in (
                            target_row.strip().strip("[").strip("]").split(", ")
                        ):
                            target_row_array.append(float(probability))
                        target.append(target_row_array)
                    targets.append(target)

            return features, targets

        # Seed RBUF with training data
        # features, targets = load_training_data("RBUF_seed/7x7-3-100.csv")
        # for feature, target in zip(features, targets):
        #     self.RBUF.append((feature, target))
        # print("RBUF loaded with", len(self.RBUF), "samples")

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
            # if i == 20:
            #     self.RBUF = self.RBUF[-1000:]
            if i % self.M == 0:
                self.MCTS.ANET.save_model("./models/" + str(i) + "episodes.pth")
            self.MCTS.reset_root()
            self.MCTS.epsilon = max(
                self.MCTS.epsilon - self.MCTS.epsilon_decay_rate, 0.0
            )

    def run_episode(self):
        while not self.MCTS.root.state.is_terminal():
            self.make_move()
            if self.visualize:
                self.MCTS.root.state.visualize()

    def make_move(self):
        new_root = self.MCTS.search()

        (features, target) = (
            self.MCTS.root.state.get_state(),
            self.get_distibution(self.MCTS.root),
        )

        def write_to_csv(features, target):
            with open("trainingdata.csv", "a", newline="") as trainingdata:
                writer = csv.writer(trainingdata)
                writer.writerow(features + target)

        write_to_csv(features, target)

        while len(self.RBUF) >= 100000:
            self.RBUF.pop(0)
        self.RBUF.append((features, target))
        self.MCTS.root = new_root
        self.MCTS.root.parent = None

    def get_distibution(self, root: Node):

        D = [[0.0] * len(root.state.board) for _ in range(len(root.state.board))]

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
                    D[i][j] = 0.0
        return D
