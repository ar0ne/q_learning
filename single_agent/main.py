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

    def __hash__(self):
        return hash((self.pos_x, self.pos_y))

    def __eq__(self, other):
        return (self.pos_x, self.pos_y) == (other.pos_x, other.pos_y)

    def __ne__(self, other):
        return not(self == other)

    def __str__(self):
        return "(%d, %d)" % (self.pos_x, self.pos_y)


class QLearning():
    def __init__(self):
        self.WIDTH = 16
        self.HEIGHT = 9

        self.Q = self.init_q()

        # Rewards matrix
        self.R = [
            [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, -100, 0, 0],
            [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, -100, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0],
            [0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0, 100]
        ]

        self.score = 10
        self.walk_reward = -0.04

        self.Gamma = .8
        self.Epochs = 10

    def init_q(self):
        q = {}
        for x in xrange(self.WIDTH):
            for y in xrange(self.HEIGHT):
                state = State(x, y)
                actions = self.get_allowed_actions(state)
                temp = {}
                for action in actions:
                    temp[action()] = 0.1
                q[state] = temp
        return q

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
        allowed = self.Q[state].items()
        return random.choice(allowed)[0]

    def calculate_q(self, state, action):
        # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
        return self.R[state.pos_y][state.pos_x] + self.Gamma * self.get_max_q(action)

    def get_max_q(self, next_state):
        allowed = self.Q[next_state].items()
        max_v = -1.
        for allowed_state in allowed:
            if allowed_state[1] > max_v:
                max_v = allowed_state[1]
        return max_v

    def default_state(self):
        return State(0, 8)

    def update_q(self, state, action, alpha, increment):
        self.Q[state][action] *= 1 - alpha
        self.Q[state][action] += alpha * increment

    def training(self):
        # init state
        state = self.default_state()
        count = 1
        alpha = 1
        tick = 1
        while count < self.Epochs:

            # print("%d : %d" % (state.pos_x, state.pos_y))

            action = self.choose_next_action(state)

            max_q = self.calculate_q(state, action)

            self.score -= self.walk_reward

            self.update_q(state, action, alpha, max_q + self.score)

            self.show_progress(state, action)

            if self.is_game_failed(action):
                # print("Failed")
                state = self.default_state()
            elif self.is_game_won(action):
                # goal completed, start next epoch
                count += 1
                state = self.default_state()
                print("Success")
            else:
                state = action

            # Update the learning rate
            alpha = pow(tick, -0.1)

    def is_game_failed(self, action):
        return self.R[action.pos_y][action.pos_x] == -100

    def is_game_won(self, action):
        return action.pos_x == 15 and action.pos_y == 8

    def normalize_q(self):
        pass

    def show_progress(self, state, new_action):
        os.system('cls' if os.name == 'nt' else 'clear')
        for i in xrange(self.HEIGHT):
            for j in xrange(self.WIDTH):
                if new_action.pos_x == j and new_action.pos_y == i:
                    print('\033[0;31;40m%6.2f' % self.Q[state][new_action], end='')
                else:
                    print('\033[1;32;40m%6d' % self.R[i][j], end='')
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
