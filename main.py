import yaml


def read_parameters_from_yaml(file_path: str):
    with open(file_path, "r") as file:
        parameters = yaml.safe_load(file)
    return parameters


# Example usage
file_path = "pivotal_parameters.yaml"
parameters = read_parameters_from_yaml(file_path)
print(parameters)

from Hex import Hex, Player
from Nim import Nim
from NeuralNet import NeuralNet
from MonteCarloTreeSearch import Node, MonteCarloTreeSearch
from Controller import Controller

# import random

# random.seed(123)

board: list[list[Player]] = []

for i in range(5):
    row: list[Player] = []
    for j in range(5):
        row.append(Player.EMPTY)
    board.append(row)


Hex = Hex(board, True)

Nim = Nim(17, 5, True)

NN = NeuralNet(len(board) ** 2 + 1, len(board) ** 2, 2, 5)

MCTS = MonteCarloTreeSearch(1, Node(Hex, None, None), NN, 1, 10)

controller = Controller(MCTS, NN)

controller.run(10)
