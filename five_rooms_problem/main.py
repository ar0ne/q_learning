#!/usr/bin/env python
from __future__ import print_function
import random

try:
    xrange
except NameError:
    xrange = range

class QLearning():
    def __init__(self):

        # Rewards matrix
        self.R = [[-1, -1, -1, -1, 0, -1],
                  [-1, -1, -1, 0, -1, 100],
                  [-1, -1, -1, 0, -1, -1],
                  [-1, 0, 0, -1, 0, -1],
                  [-1, 0, 0, -1, -1, 100],
                  [-1, 0, -1, -1, 0, 100]]

        self.Q = [[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]

        self.MaxStateCount = len(self.Q)
        self.Gamma = .5
        self.Epochs = 500

    def get_allowed_actions(self, state):
        allowed_actions = []
        for i in xrange(self.MaxStateCount):
            if self.R[state][i] >= 0:
                allowed_actions.append(i)
        return allowed_actions

    def choose_next_action(self, current_state):
        allowed = self.get_allowed_actions(current_state)
        return random.choice(allowed)

    def get_max_q(self, next_state):
        allowed = self.get_allowed_actions(next_state)
        max = 0
        for i in allowed:
            if self.Q[next_state][i] > max:
                max = self.Q[next_state][i]
        return max

    def calculate_q(self, state, action):
        # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        return self.R[state][action] + self.Gamma * self.get_max_q(action)

    def training(self):
        # get random state
        state = self.next_state()
        count = 0

        while count < self.Epochs:

            action = self.choose_next_action(state)

            self.Q[state][action] = self.calculate_q(state, action)

            if state == action:
                # goal completed, start next epoch
                count += 1
                state = self.next_state()
            else:
                state = action

    def next_state(self):
        return random.choice(range(self.MaxStateCount))

    def normalize_q(self):
        max = 0.
        for row in self.Q:
            for v in row:
                if v > max:
                    max = v

        for ndx, value in enumerate(self.Q):
            for idx, q_value in enumerate(value):
                self.Q[ndx][idx] *= 100 / max

    @staticmethod
    def print_q(q_matrix):
        for row in q_matrix:
            for val in row:
                print('%5.0f' % val, end='')
            print()

    def get_result(self, q_matrix, start_state):
        i = 0
        state = start_state
        limit = 10
        while i < limit:
            max = -1
            print("%d -> " % state, end='')
            for idx, value in enumerate(self.Q[state]):
                if value > max:
                    max = value
                    next_state = idx
            if state == next_state:
                break
            state = next_state
            i += 1


def run():
    q_learning = QLearning()

    q_learning.training()

    q_learning.normalize_q()

    q_learning.print_q(q_learning.Q)

    for i in xrange(q_learning.MaxStateCount):
        q_learning.get_result(q_learning.Q, i)
        print()


if __name__ == '__main__':
    run()

    """
    $python main.py
        0    0    0    0   50    0
        0    0    0   25    0  100
        0    0    0   25    0    0
        0   50   12    0   50    0
        0   50   12    0    0  100
        0   50    0    0   50  100
    0 -> 4 -> 5 -> 
    1 -> 5 -> 
    2 -> 3 -> 1 -> 5 -> 
    3 -> 1 -> 5 -> 
    4 -> 5 -> 
    5 -> 
    """
