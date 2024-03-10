import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int,
        neurons_per_layer: int,
        activation_function: str,
        weights: torch.Tensor | None = None,
        biases: torch.Tensor | None = None,
    ):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer

        # Choose the activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU()
        elif activation_function == "sigmoid":
            self.activation_function = nn.Sigmoid()
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
        elif activation_function == "linear":
            self.activation_function = nn.Identity()
        else:
            raise ValueError("Activation function not recognized")

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
        x = self.input_layer(x)
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        print("Model saved")

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
        print("Model loaded")
