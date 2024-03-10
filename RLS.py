from NeuralNet import NeuralNet
from HexDataset import HexDataset
import torch
from torch.utils.data import DataLoader


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
        RBUF: HexDataset,
        subset_size: int,
        epochs: int,
        learning_rate: float,
    ):
        training_data = DataLoader(RBUF, batch_size=subset_size, shuffle=True)
        size = len(training_data.dataset)
        optimizer = self.optimizer(neural_net.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        neural_net.train(True)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            for batch, (X, y) in enumerate(training_data):
                # Compute prediction and loss
                pred = neural_net(X)
                loss = criterion(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch % 1 == 0:
                    loss, current = loss.item(), batch * subset_size + len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        neural_net.train(False)
