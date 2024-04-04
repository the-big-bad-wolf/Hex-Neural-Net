import yaml
from Hex import Hex
from NeuralNet import NeuralNet
from MonteCarloTreeSearch import Node, MonteCarloTreeSearch
from Controller import Controller
from TOPP import TOPP, NNPlayer
from RLS import RLS
import matplotlib.pyplot as plt


def read_parameters_from_yaml(file_path: str):
    with open(file_path, "r") as file:
        parameters = yaml.safe_load(file)
        file.close()
    return parameters


parameters = read_parameters_from_yaml("pivotal_parameters.yaml")

MCTS_params = parameters["MCTS"]
MCTS_exploration = int(MCTS_params["MCTS_exploration"])
rollout_duration = int(MCTS_params["rollout_duration"])
epsilon = int(MCTS_params["epsilon"])

NN_params = parameters["neural_net"]
hidden_layers = int(NN_params["hidden_layers"])
neurons_per_layer = int(NN_params["neurons_per_layer"])
activation_function = NN_params["activation_function"]

training_params = parameters["training"]
optimizer = training_params["optimizer"]
epochs = int(training_params["epochs"])
episodes = int(training_params["episodes"])
learning_rate = float(training_params["learning_rate"])
RBUF_sample_size = int(training_params["RBUF_sample_size"])

hex_params = parameters["hex"]
board_size = int(hex_params["board_size"])

M = int(parameters["M"])
G = int(parameters["G"])
visualize = bool(parameters["visualize"])

board = Hex.empty_board(board_size)
Hex = Hex(board, True)

NN = NeuralNet(
    input_size=len(board) ** 2 + 1,
    output_size=len(board) ** 2,
    hidden_layers=hidden_layers,
    neurons_per_layer=neurons_per_layer,
    activation_function=activation_function,
)

MCTS = MonteCarloTreeSearch(
    exploration_param=MCTS_exploration,
    root=Node(Hex, None, None),
    ANET=NN,
    epsilon=epsilon,
    rollout_duration=rollout_duration,
)

RLS = RLS(optimizer=optimizer)

controller = Controller(
    MCTS=MCTS,
    RLS=RLS,
    RBUF_sample_size=RBUF_sample_size,
    learning_rate=learning_rate,
    training_epochs=epochs,
    M=M,
    visualize=visualize,
)

controller.run(episodes)

players: list[NNPlayer] = []
for i in range(0, episodes + 1, M):
    neural_net = NeuralNet(
        input_size=board_size**2 + 1,
        output_size=board_size**2,
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons_per_layer,
        activation_function=activation_function,
    )
    neural_net.load_model(f"./models/{i}episodes.pth")
    players.append(NNPlayer(str(i), neural_net))

tournament = TOPP(board_size, players, visualize)
results = tournament.round_robin(G)

# Plotting the results in a bar chart
x = list(results.keys())
y = list(results.values())

plt.bar(x, y)
plt.xlabel("Player")
plt.ylabel("Wins")
plt.title("Tournament Results")
plt.xticks(x, [player.id for player in players])
plt.show()
