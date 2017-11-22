#!/usr/bin/env python
from __future__ import print_function
import random
import time
import os


class State():
    def __init__(self, pos_x=0, pos_y=0, action=''):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.action = action

    def move_left(self):
        return State(self.pos_x, self.pos_y - 1, 'left')

    def move_right(self):
        return State(self.pos_x, self.pos_y + 1, 'right')

    def move_top(self):
        return State(self.pos_x - 1, self.pos_y, 'top')

    def move_bottom(self):
        return State(self.pos_x + 1, self.pos_y, 'bottom')

    def __hash__(self):
        return hash((self.pos_x, self.pos_y))

    def __eq__(self, other):
        return (self.pos_x, self.pos_y) == (other.pos_x, other.pos_y)

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "(%d, %d) - %s" % (self.pos_x, self.pos_y, self.action)


class QLearning():
    def __init__(self):
        # self.WIDTH = 16
        # self.HEIGHT = 9
        self.WALK_REWARDS = -0.04
        self.GAMMA = .8
        self.EPSILON = 0.1
        self.EPOCHS = 100
        self.INIT_Q_VALUE = 0  # in most cases should be zero
        self.ALPHA = 0.1

        self.FRAME_RATE = 0.2

        self.WIDTH = 8
        self.HEIGHT = 9

        # self.R = [
        #     [0, 0, 0, 0, 0, 0, -100, 0],
        #     [0, 0, 0, 0, 0, 0, -100, 0],
        #     [0, 0, 0, 0, 0, 0, -100, 0],
        #     [0, 0, 0, 0, 0, 0, -100, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 100],
        #     [0, 0, -100, -100, -100, -100, -100, -100]
        # ]

        self.R = [
            [-1,-1,-1,-1,-1,-1, -100,-1],
            [-1,-1,-1,-1,-1,-1, -100,-1],
            [-1,-1,-1,-1,-1,-1, -100,-1],
            [-1,-1,-1,-1,-1,-1, -100,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1, 100],
            [-1,-1, -100, -100, -100, -100, -100, -100]
        ]

        self.score = 0
        self.success = 0
        self.failures = 0

        self.start_state = State(0, 8)
        self.end_state = State(7, 7)

        # Init Q matrix
        self.Q = self.init_q()

        # Rewards matrix
        # self.R = [
        #     [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, -100, 0, 0],
        #     [0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, -100, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0],
        #     [0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0, 100]
        # ]



    def init_q(self):
        q = {}
        for x in xrange(self.WIDTH):
            for y in xrange(self.HEIGHT):
                state = State(x, y)
                actions = self.get_allowed_actions(state)
                temp = {}
                for action in actions:
                    temp[action()] = self.INIT_Q_VALUE
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

    def choose_next_action(self, state, randomly=True):
        allowed = self.Q[state].items()
        if randomly and random.random() < self.EPSILON:
            return random.choice(allowed)[0]
        else:
            q = [a[1] for a in allowed]
            max_q = max(q)
            count = q.count(max_q)
            if count > 1:
                best = [a for a in allowed if a[1] == max_q]
                return random.choice(best)[0]
            else:
                return [a[0] for a in allowed if a[1] == max_q][0]

    def get_max_q(self, next_state):
        allowed = self.Q[next_state].items()
        max_v = -10000
        for allowed_state in allowed:
            if allowed_state[1] > max_v:
                max_v = allowed_state[1]
        return max_v

    def learn(self, state, next_state, reward):
        max_q = self.get_max_q(next_state)
        self.learnQ(state, next_state, reward, reward + self.GAMMA * max_q)

    def learnQ(self, state, action, reward, value):
        old_q_value = self.Q[state][action]
        self.Q[state][action] = old_q_value + self.ALPHA * (value - old_q_value)

    def training(self):
        # init state
        state = self.start_state

        count = 0
        reward = 0
        while count < self.EPOCHS:

            next_state = self.choose_next_action(state)

            # self.show_progress(state, next_state)

            if self.is_game_failed(next_state):
                self.failures += 1
                reward = -100
                self.learn(state, next_state, reward)
                state = self.start_state
                continue

            elif self.is_game_won(next_state):
                # goal completed, start next epoch
                count += 1
                self.success += 1
                reward = 50
                self.learn(state, next_state, reward)
                state = self.start_state
                continue

            self.learn(state, next_state, reward)
            state = next_state

            # Update the learning rate
            # alpha = pow(tick, -0.1)

    def is_game_failed(self, action):
        return self.R[action.pos_y][action.pos_x] == -100

    def is_game_won(self, action):
        return action.pos_x == self.end_state.pos_x and action.pos_y == self.end_state.pos_y

    def normalize_q(self):
        pass

    def show_progress(self, state, new_action):
        os.system('cls' if os.name == 'nt' else 'clear')
        for i in xrange(self.HEIGHT):
            for j in xrange(self.WIDTH):
                if new_action.pos_x == j and new_action.pos_y == i:
                    print('\033[0;31;40m%6.2f' % self.Q[state][new_action], end='')
                else:
                    if self.R[i][j] == -100:
                        print('\033[1;32;40m%6s' % "x", end='')
                    elif self.R[i][j] == 100:
                        print('\033[1;32;40m%6s' % "[]", end='')
                    else:
                        print('\033[1;32;40m%6s' % ".", end='')
            print()
        print()
        print("Success: %d, Failures: %d" % (self.success, self.failures))
        [print("%2.2f -- %s" % (i[1], i[0])) for i in self.Q[state].items()]
        time.sleep(self.FRAME_RATE)


    def show_final_result(self):
        state = self.start_state
        steps = 0
        while True:
            next_state = self.choose_next_action(state, False)
            self.show_progress(state, next_state)
            if self.is_game_won(next_state):
                print("Steps: %d" % steps)
                break
            state = next_state
            steps += 1

def run():
    q_learning = QLearning()
    q_learning.training()
    print("Success: %d, Failures: %d" % (q_learning.success, q_learning.failures))
    # q_learning.normalize_q()
    q_learning.show_final_result()
    # q_learning.print_q(q_learning.Q)


if __name__ == '__main__':
    run()
