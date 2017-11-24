#!/usr/bin/env python
from __future__ import print_function
import random
import time
import os


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
        allowed = [state.move_none]
        if self.is_movable_to_the_left(state):
            allowed.append(state.move_left)
        if self.is_movable_to_the_right(state):
            allowed.append(state.move_right)
        if self.is_movable_to_the_top(state):
            allowed.append(state.move_top)
        if self.is_movable_to_the_bottom(state):
            allowed.append(state.move_bottom)
        return allowed

    def add_agent1(self, actor):
        self.agent1 = actor
        self.q1 = self.init_q(actor.shift)

    def add_agent2(self, actor):
        self.agent2 = actor
        self.q2 = self.init_q(actor.shift)

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

    def is_game_failed(self, state):
        return self.ROOM[state.y][state.x] == -100

    def is_game_won(self, state):
        return self.ROOM[state.y][state.x] == 100

    def training(self):
        count = 0
        tick = 1
        alpha = 1
        st1 = self.agent1
        st2 = self.agent2
        st1_action = self.choose_next_action(st1, st2, self.q1)
        st2_action = self.choose_next_action(st2, st1, self.q2)

        while count < self.EPOCHS:

            reward = self.WALK_REWARDS

            # self.show_progress(state, action)

            old_st1 = st1
            old_st2 = st2
            old_st1_action = st1_action
            old_st2_action = st2_action

            if self.is_game_failed(st1_action) or self.is_game_failed(st2_action):
                reward = -100
                self.failures += 1
                next_st1_action = self.agent1
                next_st2_action = self.agent2
            elif self.is_game_won(st1_action) or self.is_game_won(st2_action):
                reward = 100
                self.success += 1
                count += 1
                next_st1_action = self.agent1
                next_st2_action = self.agent2
            else:
                next_st1_action = st1_action
                next_st2_action = st2_action

            st1_updated_q = self.get_updated_q(old_st1, old_st2, self.q1, st1_action, alpha, reward, st1, st2)
            st2_updated_q = self.get_updated_q(old_st2, old_st1, self.q2, st2_action, alpha, reward, st2, st1)

            self.q1[old_st1][old_st2][old_st1_action] = st1_updated_q
            self.q1[old_st2][old_st1][old_st2_action] = st2_updated_q

            st1 = next_st1_action
            st1_action = self.choose_next_action(st1, st2, self.q1)

            st2 = next_st2_action
            st2_action = self.choose_next_action(st2, st1, self.q2)


            # Update the learning rate
            alpha = pow(tick, -0.1)
            tick += 1



def run():
    agent1 = State(0, 8, shift=1)
    agent2 = State(1, 8, shift=2)

    q_learn = QLearning()

    q_learn.add_agent1(agent1)
    q_learn.add_agent2(agent2)

    q_learn.training()


if __name__ == '__main__':
    run()
