#!/usr/bin/env python
from __future__ import print_function
import random
import time
import os

class State():
    def __init__(self, pos_x=0, pos_y=0):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def move_left(self):
        return State(self.pos_x, self.pos_y - 1)

    def move_right(self):
        return State(self.pos_x, self.pos_y + 1)

    def move_top(self):
        return State(self.pos_x - 1, self.pos_y)

    def move_bottom(self):
        return State(self.pos_x + 1, self.pos_y)


class QLearning():
    def __init__(self):
        self.WIDTH = 16
        self.HEIGHT = 9

        self.Q = [[0 for x in range(self.WIDTH)] for y in range(self.HEIGHT)]

        # Rewards matrix
        # self.R = [
        #     [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -100, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -100, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
        #     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
        #     [-1, -1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1, 100]
        # ]
        self.R = [
            [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, -100, 0, -1],
            [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, -100, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, -1],
            [0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0, 100]
        ]


        self.Gamma = .8
        self.Epochs = 10

    def is_moveable_to_the_left(self, state):
        return state.pos_y > 0

    def is_moveable_to_the_right(self, state):
        return state.pos_y < self.HEIGHT - 1

    def is_moveable_to_the_top(self, state):
        return state.pos_x > 0

    def is_moveable_to_the_bottom(self, state):
        return state.pos_x < self.WIDTH - 1

    def get_allowed_actions(self, state):
        allowed = []
        if self.is_moveable_to_the_left(state):
            allowed.append(state.move_left)
        if self.is_moveable_to_the_right(state):
            allowed.append(state.move_right)
        if self.is_moveable_to_the_top(state):
            allowed.append(state.move_top)
        if self.is_moveable_to_the_bottom(state):
            allowed.append(state.move_bottom)
        return allowed

    def choose_next_action(self, state):
        allowed = self.get_allowed_actions(state)
        return random.choice(allowed)()

    def calculate_q(self, state, action):
        # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        return self.R[state.pos_y][state.pos_x] + self.Gamma * self.get_max_q(action)

    def get_max_q(self, next_state):
        allowed = self.get_allowed_actions(next_state)
        max = 0
        for action in allowed:
            allowed_state = action()
            if self.Q[allowed_state.pos_y][allowed_state.pos_x] > max:
                max = self.Q[allowed_state.pos_y][allowed_state.pos_x]
        return max

    def default_state(self):
        return State(0, 8)

    def training(self):
        # init state
        state = self.default_state()
        count = 0

        while count < self.Epochs:

            # print("%d : %d" % (state.pos_x, state.pos_y))

            action = self.choose_next_action(state)

            self.Q[state.pos_y][state.pos_x] = self.calculate_q(state, action)

            self.show_progress(self.Q, state)

            if self.R[action.pos_y][action.pos_x] == -100:
                # print("Failed")
                state = self.default_state()
            elif action.pos_x == 15 and action.pos_y == 8:
                # goal completed, start next epoch
                count += 1
                state = self.default_state()
                print("Success")
            else:
                state = action

    def normalize_q(self):
        pass

    @staticmethod
    def show_progress(q_matrix, new_action):
        os.system('cls' if os.name == 'nt' else 'clear')
        for rdx, row in enumerate(q_matrix):
            for cdx, val in enumerate(row):
                if new_action.pos_x == cdx and new_action.pos_y == rdx:
                    print('\033[0;31;40m%6.2f' % val, end='')
                else:
                    print('\033[1;32;40m%6.2f' % val, end='')
            print()
        print()
        time.sleep(0.1)


def run():
    q_learning = QLearning()
    q_learning.training()
    q_learning.normalize_q()
    # q_learning.print_q(q_learning.Q)


if __name__ == '__main__':
    run()
