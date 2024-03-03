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
        activation_function: str,
        optimizer: str,
        weights=None,
        biases=None,
    ):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer

        # Choose the activation function
        if activation_function == "relu":
            self.activation_function = torch.relu
        elif activation_function == "sigmoid":
            self.activation_function = torch.sigmoid
        elif activation_function == "tanh":
            self.activation_function = torch.tanh
        elif activation_function == "linear":
            self.activation_function = lambda x: x
        else:
            raise ValueError("Activation function not recognized")

        # Choose optimizer
        if optimizer == "adam":
            self.optimizer = optim.Adam
        elif optimizer == "sgd":
            self.optimizer = optim.SGD
        elif optimizer == "adagrad":
            self.optimizer = optim.Adagrad
        elif optimizer == "RMSPROP":
            self.optimizer = optim.RMSprop
        else:
            raise ValueError("Optimizer not recognized")

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
        x = self.activation_function(self.input_layer(x.float()))
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        x = torch.nn.functional.softmax(x, dim=0)
        return x

    def update_params(
        self,
        RBUF: list[tuple[list, list[list[float]]]],
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

        optimizer = self.optimizer(self.parameters(), lr=learning_rate)

        print("input_tensor", input_tensor)
        print("target_tensor", target_tensor)
        for _ in range(epochs):
            losses = []
            optimizer.zero_grad()
            outputs = self.forward(input_tensor)
            loss = torch.nn.CrossEntropyLoss()(outputs, target_tensor)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            print(f"Loss: {sum(losses) / len(losses)}")
