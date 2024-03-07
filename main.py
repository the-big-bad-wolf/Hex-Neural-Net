import yaml
from Hex import Hex
from NeuralNet import NeuralNet
from MonteCarloTreeSearch import Node, MonteCarloTreeSearch
from Controller import Controller
from TOPP import TOPP


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
nr_games = int(parameters["nr_games"])

board = Hex.empty_board(board_size)
Hex = Hex(board, True)

NN = NeuralNet(
    input_size=len(board) ** 2 + 1,
    output_size=len(board) ** 2,
    hidden_layers=hidden_layers,
    neurons_per_layer=neurons_per_layer,
    activation_function=activation_function,
    optimizer=optimizer,
)

MCTS = MonteCarloTreeSearch(
    exploration_param=MCTS_exploration,
    root=Node(Hex, None, None),
    ANET=NN,
    epsilon=epsilon,
    rollout_duration=rollout_duration,
)

controller = Controller(
    MCTS,
    learning_rate=learning_rate,
    RBUF_sample_size=RBUF_sample_size,
    training_epochs=epochs,
)

controller.run(episodes)


NN1 = NeuralNet(
    input_size=len(board) ** 2 + 1,
    output_size=len(board) ** 2,
    hidden_layers=hidden_layers,
    neurons_per_layer=neurons_per_layer,
    activation_function=activation_function,
    optimizer=optimizer,
)
NN2 = NeuralNet(
    input_size=len(board) ** 2 + 1,
    output_size=len(board) ** 2,
    hidden_layers=hidden_layers,
    neurons_per_layer=neurons_per_layer,
    activation_function=activation_function,
    optimizer=optimizer,
)
NN1.load_model("0episodes.pth")
NN2.load_model("50episodes.pth")

tournament = TOPP(board_size, [NN1, NN2])

wins = 0
for _ in range(nr_games):
    result = tournament.run_match((NN1, NN2))
    if result == 1:
        wins += 1

print(f"NN1 won {wins/nr_games}% of games")
