from Hex import Hex, Player
from NeuralNet import NeuralNet
import torch
import copy


class TOPP:
    def __init__(self, board_size: int, NNplayers: list[NeuralNet]) -> None:
        board: list[list[Player]] = []
        for _ in range(board_size):
            row: list[Player] = []
            for _ in range(board_size):
                row.append(Player.EMPTY)
            board.append(row)
        self.board = board
        self.game = Hex(board, True)
        self.NNplayers = NNplayers

    def run_match(self, players: tuple[NeuralNet, NeuralNet]):
        game = copy.deepcopy(self.game)
        current_player = players[0]
        while not game.is_terminal():
            distribution = current_player(torch.tensor(game.get_state()))
            legal_actions = game.get_legal_actions()
            distribution = distribution.reshape(len(game.board), len(game.board))
            for i in range(len(distribution)):
                for j in range(len(distribution[i])):
                    if (i, j) not in legal_actions:
                        distribution[i][j] = float("-inf")
            # Get the indices of the top 3 highest values in the distribution
            top_indices = torch.topk(distribution.flatten(), k=3).indices
            top_probabilities = torch.topk(distribution.flatten(), k=3).values
            top_probabilities = torch.nn.functional.softmax(top_probabilities, dim=0)
            # Generate a random number between 0 and 1
            random_number = torch.rand(1).item()
            if random_number < top_probabilities[0]:
                move = top_indices[0]
            elif random_number < top_probabilities[0] + top_probabilities[1]:
                move = top_indices[1]
            else:
                move = top_indices[2]

            row = move // len(game.board)
            col = move % len(game.board)

            game = game.take_action((int(row), int(col)))

            current_player = players[1] if current_player == players[0] else players[0]
        game.visualize()
        return game.get_result()
