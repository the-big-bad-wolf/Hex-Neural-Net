from NeuralNet import NeuralNet
from Hex import Hex, Player
import torch
from ActorClient import ActorClient
import numpy as np

actor = NeuralNet(50, 49, 2, 50, "tanh")
actor.load_state_dict(torch.load("models/50episodes.pth"))


class MyClient(ActorClient):
    def handle_get_action(self, state):
        state = state[1:] + [state[0]]  # put player-turn in start of state
        for i in range(len(state)):
            if state[i] == 2:
                state[i] = -1
        for i in range(len(state)):
            match state[i]:
                case 0:
                    state[i] = Player.EMPTY
                case 1:
                    state[i] = Player.PLAYER1
                case -1:
                    state[i] = Player.PLAYER2

        player1_turn = False
        if state[-1] == 1:
            player1_turn = True

        np_state = np.reshape(state[:-1], (7, 7))
        np_state = np_state.tolist()

        game = Hex(np_state, player1_turn)
        float_state = [float(cell.value) for cell in state]
        distribution = actor(torch.tensor(float_state))
        distribution = distribution.reshape(len(game.board), len(game.board))
        legal_actions = game.get_legal_actions()
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

        return int(row), int(col)

    def handle_series_start(
        self, unique_id, series_id, player_map, num_games, game_params
    ):
        print("Starting series", series_id, "with", num_games, "games")
        print("Player map:", player_map)
        print("Game params:", game_params)
        print("Unique ID:", unique_id)
        return

    def handle_series_end(self, series_id, results):
        print("Series", series_id, "ended with results:", results)
        return

    def handle_tournament_over(self, score):
        print("Tournament over with score:", score)


# Initialize and run your overridden client when the script is executed
if __name__ == "__main__":
    client = MyClient()
    client.run()
