#!/usr/bin/env python
from __future__ import print_function
import random
import time
import os
import time


class State:
    def __init__(self, x=0, y=0, movement='', shift=1):
        self.x = x
        self.y = y
        self.shift = shift
        self.movement = movement

    def move_left(self):
        return State(self.x - 1, self.y, 'left', shift=self.shift)

    def move_right(self):
        return State(self.x + 1, self.y, 'right', shift=self.shift)

    def move_top(self):
        return State(self.x, self.y - 1, 'top', shift=self.shift)

    def move_bottom(self):
        return State(self.x, self.y + 1, 'bottom', shift=self.shift)

    def move_none(self):
        return State(self.x, self.y, 'none', shift=self.shift)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "%d: (%d, %d) - %s" % (self.shift, self.x, self.y, self.movement)


class QLearning:
    def __init__(self):
        self.GAMMA = .8
        self.EPSILON = 0.2
        self.EPOCHS = 100
        self.INIT_Q_VALUE = 0  # in most cases should be zero
        self.ALPHA = 0.1
        self.FRAME_RATE = 0.15
        self.WALK_REWARDS = -0.1  # IMPORTANT TO HAVE IT IN RANGE -0.3 .. 0

        self.success = 0
        self.failures = 0

        # self.actors = []

        self.ROOM = [
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

        self.HEIGHT = len(self.ROOM)
        self.WIDTH = len(self.ROOM[0])

        self.q1 = {}
        self.q2 = {}

        self.agent1 = None
        self.agent2 = None

    def init_q(self, shift):
        Q = {}
        for x in xrange(self.WIDTH):
            for y in xrange(self.HEIGHT):
                agent1_state = State(x, y, shift=shift)
                agent1_allowed_actions = self.get_allowed_actions(agent1_state)
                Q[agent1_state] = {}
                agent2_possible_states = self.get_possible_states_of_partner(agent1_state)
                for agent2_state in agent2_possible_states:
                    temp = {}
                    for action in agent1_allowed_actions:
                        temp[action()] = self.INIT_Q_VALUE
                    Q[agent1_state][agent2_state] = temp
        return Q

    def get_possible_states_of_partner(self, agent, limit=3):
        possible_states = []
        for i in xrange(-limit + agent.x, limit + 1 + agent.x):
            for j in xrange(-limit + agent.y, limit + 1 + agent.y):
                possible_state = State(i, j, shift=agent.shift)
                if self.is_it_possible_position(possible_state):
                    possible_states.append(possible_state)
        return possible_states

    def is_it_possible_position(self, state):
        return 0 <= state.x < self.WIDTH and 0 <= state.y < self.HEIGHT

    def is_movable_to_the_left(self, agent):
        return agent.x > 0

    def is_movable_to_the_right(self, agent):
        return agent.x < self.WIDTH - agent.shift

    def is_movable_to_the_top(self, agent):
        return agent.y > 0

    def is_movable_to_the_bottom(self, agent):
        return agent.y < self.HEIGHT - agent.shift

    def get_allowed_actions(self, state):
        # allowed = [state.move_none]
        allowed = []
        if self.is_movable_to_the_left(state):
            allowed.append(state.move_left)
        if self.is_movable_to_the_right(state):
            allowed.append(state.move_right)
        if self.is_movable_to_the_top(state):
            allowed.append(state.move_top)
        if self.is_movable_to_the_bottom(state):
            allowed.append(state.move_bottom)
        return allowed

    def add_agent1(self, actor, max_shift):
        self.agent1 = actor


    def add_agent2(self, actor, max_shift):
        self.agent2 = actor
        self.q2 = self.init_q(max_shift)

    def choose_next_action(self, agent_a, agent_b, agent_a_Q, randomly=True):
        allowed = agent_a_Q[agent_a][agent_b].items()
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

    def get_max_q(self, Q, st1, st2):
        allowed = Q[st1][st2].items()
        max_v = -1000
        for allowed_state in allowed:
            if allowed_state[1] > max_v:
                max_v = allowed_state[1]
        return max_v

    def get_updated_q(self, st1, st2, Q, action, alpha, r, next_st1, next_st2):
        #  Q[s',a'] = Q[s',a'] + alpha * (reward + gamma * MAX(Q,s) - Q[s',a'])
        return Q[st1][st2][action] + alpha * (r + self.GAMMA * self.get_max_q(Q, next_st1, next_st2) - Q[st1][st2][action])

    def is_game_failed(self, st1, st2):
        if self.ROOM[st1.y][st1.x] == -100 or self.ROOM[st2.y][st2.x]:
            return True
        return (pow(st1.x - st2.x, 2) + pow(st1.y - st2.y, 2)) ** .5 > 3

    def is_game_won(self, st1, st2):
        return self.ROOM[st1.y][st1.x] == 100 or self.ROOM[st2.y][st2.x] == 100

    def training(self, start_st1, start_st2):
        count = 0
        tick = 1
        alpha = 1
        st1 = start_st1
        st2 = start_st2
        st1_action = self.choose_next_action(st1, st2, self.q1)
        st2_action = self.choose_next_action(st2, st1, self.q2)

        while count < self.EPOCHS:

            reward = self.WALK_REWARDS

            # self.show_progress(st1, st2, st1_action, st2_action, self.q1, self.q2)

            old_st1 = st1
            old_st2 = st2
            old_st1_action = st1_action
            old_st2_action = st2_action

            if self.is_game_failed(st1_action, st2_action):
                reward = -100
                self.failures += 1
                next_st1_action = start_st1
                next_st2_action = start_st2
            elif self.is_game_won(st1_action, st2_action):
                reward = 100
                self.success += 1
                count += 1
                next_st1_action = start_st1
                next_st2_action = start_st2
            else:
                next_st1_action = st1_action
                next_st2_action = st2_action

            st1_updated_q = self.get_updated_q(old_st1, old_st2, self.q1, st1_action, alpha, reward, st1, st2)
            st2_updated_q = self.get_updated_q(old_st2, old_st1, self.q2, st2_action, alpha, reward, st2, st1)

            self.q1[old_st1][old_st2][old_st1_action] = st1_updated_q
            self.q1[old_st2][old_st1][old_st2_action] = st2_updated_q

            st1 = next_st1_action
            st2 = next_st2_action

            st1_action = self.choose_next_action(st1, st2, self.q1)
            st2_action = self.choose_next_action(st2, st1, self.q2)

            # Update the learning rate
            alpha = pow(tick, -0.1)
            tick += 1

    def show_progress(self, st1, st2, st1_action, st2_action, q1, q2):
        os.system('cls' if os.name == 'nt' else 'clear')
        for i in xrange(self.HEIGHT):
            for j in xrange(self.WIDTH):
                if st1_action.x == j and st1_action.y == i:
                    print('\033[0;34;40m%6.2f' % q1[st1][st2][st1_action], end='')
                elif st2_action.x == j and st2_action.y == i:
                    print('\033[0;33;40m%6.2f' % q2[st2][st1][st2_action], end='')
                else:
                    if self.ROOM[i][j] == -100:
                        print('\033[1;31;40m%6s' % "x", end='')
                    elif self.ROOM[i][j] == 100:
                        print('\033[1;32;40m%6s' % "[]", end='')
                    else:
                        print('\033[1;32;40m%6s' % ".", end='')
            print()
        print()
        print("\033[1;32;40mSuccess: %d, Failures: %d" % (self.success, self.failures))
        # [print("%2.2f -- %s" % (i[1], i[0])) for i in self.Q[state].items()]
        time.sleep(self.FRAME_RATE)

    def show_final_result(self, st1, st2, q1, q2):
        steps = 0
        while True:
            next_st1 = self.choose_next_action(st1, st2, q1, False)
            next_st2 = self.choose_next_action(st2, st1, q2, False)
            self.show_progress(st1, st2, next_st1, next_st2, self.q1, self.q2)
            if self.is_game_won(next_st1, next_st2):
                print("Steps: %d" % steps)
                break
            st1 = next_st1
            st2 = next_st2
            steps += 1


def run():
    q_learn = QLearning()

    agent1 = State(0, 8, shift=1)
    agent2 = State(1, 8, shift=2)

    max_shift = max(agent1.shift, agent2.shift)
    q_learn.q1 = q_learn.init_q(max_shift)
    q_learn.q2 = q_learn.init_q(max_shift)

    start_time = time.time()
    q_learn.training(agent1, agent2)
    print("--- %s seconds ---" % (time.time() - start_time))
    input("show final result")

    q_learn.show_final_result(agent1, agent2, q_learn.q1, q_learn.q2)


if __name__ == '__main__':
    run()
