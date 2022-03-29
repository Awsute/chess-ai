from neuralnet import *
import chess

class Node:
    def __init__(self, state, value : float):
        self.state = state
        self.value = value
        self.num_visits = 0
        self.children - []