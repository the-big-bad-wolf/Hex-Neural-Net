import torch
import torch.nn as nn
import torch.optim as optim
import random


class NeuralNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int,
        neurons_per_layer: int,
        weights=None,
        biases=None,
    ):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer

        # Create the input layer
        self.input_layer = nn.Linear(input_size, neurons_per_layer)
        if weights is not None and biases is not None:
            self.input_layer.weight = nn.Parameter(weights[0])
            self.input_layer.bias = nn.Parameter(biases[0])

        # Create the hidden layers
        self.hidden = nn.ModuleList()
        for i in range(hidden_layers):
            layer = nn.Linear(neurons_per_layer, neurons_per_layer)
            if weights is not None and biases is not None:
                layer.weight = nn.Parameter(weights[i + 1])
                layer.bias = nn.Parameter(biases[i + 1])
            self.hidden.append(layer)

        # Create the output layer
        self.output_layer = nn.Linear(neurons_per_layer, output_size)
        if weights is not None and biases is not None:
            self.output_layer.weight = nn.Parameter(weights[-1])
            self.output_layer.bias = nn.Parameter(biases[-1])

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.input_layer(x.float()))
        for layer in self.hidden:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

    def update_params(
        self,
        RBUF: list[tuple[list, list[list[float]]]],
        subset_size: int,
        epochs: int,
        learning_rate: float,
    ):

        random_subset = random.sample(RBUF, k=subset_size)
        input = [item[0] for item in random_subset]
        target = [item[1] for item in random_subset]

        input_tensor = torch.tensor(input)
        target_tensor = torch.tensor(target)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for _ in range(epochs):
            losses = []
            for i in range(subset_size):
                optimizer.zero_grad()
                outputs = self.forward(input_tensor[i])
                loss = torch.nn.functional.mse_loss(outputs, target_tensor[i].flatten())
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
            print(f"Loss: {sum(losses) / len(losses)}")
