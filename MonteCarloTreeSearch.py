from __future__ import annotations
from math import log, sqrt
from State import State
from NeuralNet import NeuralNet
import torch
import random


class MonteCarloTreeSearch:
    def __init__(
        self,
        exploration_param: float,
        root: Node,
        ANET: NeuralNet,
        epsilon: float,
        nr_rollouts: int,
    ):
        self.ANET = ANET
        self.root = root
        self.epsilon = epsilon
        self.exploration_param = exploration_param
        self.nr_rollouts = nr_rollouts

    def search(self):
        # Running the Monte Carlo Tree Search algorithm for a number of iterations
        self.root.node_expansion()
        for _ in range(self.nr_rollouts):
            leaf_node = self.traverse_tree()

            if leaf_node.visits == 0:
                result = leaf_node.rollout(self.ANET)
                leaf_node.backpropagate(result)
            else:
                leaf_node.node_expansion()
                if len(leaf_node.children) != 0:
                    random_index = random.randint(0, len(leaf_node.children) - 1)
                    random_child = leaf_node.children[random_index]
                    result = random_child.rollout(self.ANET)
                    random_child.backpropagate(result)
                else:
                    result = leaf_node.rollout(self.ANET)
                    leaf_node.backpropagate(result)
        return max(self.root.children, key=lambda x: x.visits)

    def traverse_tree(self):
        # Traversing the tree from the root to a leaf node using the tree policy
        current_node = self.root
        while current_node.is_fully_expanded() and not current_node.state.is_terminal():
            current_node = current_node.best_child(self.exploration_param)
        return current_node


class Node:
    def __init__(
        self, state: State, parent: Node | None, arcTo: tuple[int, int] | None
    ):
        self.state = state
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.sum_value = 0
        self.arcTo = arcTo

    def is_fully_expanded(self):
        # Checking if all children of a node have been generated
        if len(self.children) == len(self.state.get_legal_actions()):
            return True
        else:
            return False

    def node_expansion(self):
        # Generating child states of a parent state and connecting parent node to child nodes
        if self.is_fully_expanded():
            return
        legal_actions = self.state.get_legal_actions()
        for action in legal_actions:
            child_state = self.state.take_action(action)
            child_node = Node(child_state, self, action)
            self.children.append(child_node)

    def best_child(self, exploration_param: float):
        best_child = self.children[0]
        if self.state.player1_turn:
            for child in self.children:
                if self.UCB1(child, exploration_param) > self.UCB1(
                    best_child, exploration_param
                ):
                    best_child = child

            return best_child
        else:
            for child in self.children:
                if self.UCB1(child, exploration_param) < self.UCB1(
                    best_child, exploration_param
                ):
                    best_child = child

            return best_child

    def rollout(self, ANET: NeuralNet):
        # Simulating a random game from the state of a node to a terminal state
        current_state = self.state
        while not current_state.is_terminal():
            distribution = ANET.forward(torch.tensor(current_state.get_state()))
            legal_actions = current_state.get_legal_actions()
            distribution = distribution.reshape(
                len(current_state.board), len(current_state.board)
            )

            for i in range(len(distribution)):
                for j in range(len(distribution[i])):
                    if (i, j) not in legal_actions:
                        distribution[i][j] = float("-inf")

            row, col = divmod(distribution.argmax().item(), distribution.shape[1])
            current_state = current_state.take_action((int(row), int(col)))

            # current_state = current_state.take_action(
            #     (
            #         int(distribution.argmax().item()) + 1,
            #         int(distribution.argmax().item()) + 1,
            #     )
            # )

        return current_state.get_result()

    def backpropagate(self, result: float):
        # Passing the result of a rollout back up the tree
        self.sum_value += result
        self.visits += 1
        if self.parent is None:
            return
        self.parent.backpropagate(result)

    def UCB1(self, node: Node, exploration_param: float):
        # Calculating the value of the UCB for a node
        if node.visits == 0:
            return -float("inf") if node.state.player1_turn else float("inf")
        average_value = node.sum_value / node.visits
        assert node.parent != None
        exploration_value = exploration_param * sqrt(
            (log(node.parent.visits) / (1 + node.visits))
        )
        if not node.state.player1_turn:
            return average_value + exploration_value
        else:
            return average_value - exploration_value

    def __repr__(self):
        return f"Node({self.state})"
