from Hex import Hex, Player
from NeuralNet import NeuralNet
import torch
import copy


class NNPlayer:
    def __init__(self, id: str, neural_net: NeuralNet) -> None:
        self.id = id
        self.neural_net = neural_net


class TOPP:
    def __init__(
        self, board_size: int, NNplayers: list[NNPlayer], visualize: bool
    ) -> None:
        self.NNplayers = NNplayers
        self.board_size = board_size
        self.visualize = visualize

    def round_robin(self, nr_games: int) -> dict[str, int]:
        results: dict[str, int] = {}
        for player in self.NNplayers:
            results[player.id] = 0
        for i in range(len(self.NNplayers)):
            for j in range(i + 1, len(self.NNplayers)):
                player1 = self.NNplayers[i]
                player2 = self.NNplayers[j]
                for k in range(nr_games):
                    self.board = Hex.empty_board(self.board_size)
                    self.game = Hex(self.board, True)
                    if k % 2 == 0:
                        result = self.run_match(
                            (player1.neural_net, player2.neural_net)
                        )
                        if result == 1:
                            results[player1.id] += 1
                        else:
                            results[player2.id] += 1
                    else:
                        result = self.run_match(
                            (player2.neural_net, player1.neural_net)
                        )
                        if result == -1:
                            results[player1.id] += 1
                        else:
                            results[player2.id] += 1
        return results

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
        if self.visualize:
            game.visualize()
        return game.get_result()
