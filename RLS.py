from NeuralNet import NeuralNet
import random
import torch


class RLS:
    def __init__(
        self,
        optimizer: str,
    ):
        # Choose optimizer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD
        elif optimizer == "adagrad":
            self.optimizer = torch.optim.Adagrad
        elif optimizer == "RMSPROP":
            self.optimizer = torch.optim.RMSprop
        else:
            raise ValueError("Optimizer not recognized")
        pass

    def train(
        self,
        neural_net: NeuralNet,
        RBUF: list[tuple[list[float], list[list[float]]]],
        subset_size: int,
        epochs: int,
        learning_rate: float,
    ):
        subset_size = min(subset_size, len(RBUF))
        random_subset = random.sample(RBUF, k=subset_size)
        input = [item[0] for item in random_subset]
        target = [item[1] for item in random_subset]

        input_tensor = torch.tensor(input)
        target_tensor = torch.tensor(target)
        target_tensor = target_tensor.flatten(start_dim=1, end_dim=2)

        optimizer = self.optimizer(neural_net.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        losses = []
        neural_net.train(True)
        for _ in range(epochs):
            outputs = neural_net(input_tensor)
            loss = criterion(outputs, target_tensor)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss}")
        neural_net.train(False)
