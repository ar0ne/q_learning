#!/usr/bin/env python
from __future__ import print_function
import random
import os
import time
import sys
# import matplotlib.pyplot as plt

try:
    xrange
except NameError:
    xrange = range

def timer(fn):
    def wrapped(*args, **kwargs):
        start_time = time.time()
        fn(*args, **kwargs)
        print("\n--- %s seconds ---" % (time.time() - start_time))

    return wrapped


class State:
    def __init__(self, x=0, y=0, movement='', shift=1):
        self.x = x
        self.y = y
        self.shift = shift
        self.movement = movement

    def move_left(self):
        return State(self.x - self.shift, self.y, 'left', shift=self.shift)

    def move_right(self):
        return State(self.x + self.shift, self.y, 'right', shift=self.shift)

    def move_top(self):
        return State(self.x, self.y - self.shift, 'top', shift=self.shift)

    def move_bottom(self):
        return State(self.x, self.y + self.shift, 'bottom', shift=self.shift)

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
        self.GAMMA = 0.9
        self.EPSILON = 0.3
        self.EPOCHS = 500
        self.INIT_Q_VALUE = 0
        self.FRAME_RATE = 0.1
        self.WALK_REWARDS = -0.1
        self.MAX_ITERATIONS = 100000
        self.DEATH = -100
        self.WIN = 100

        self.success = 0
        self.failures = 0
        self.tick = 1

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

        self.statistics = {'Q1': [], 'Q2': [], 'r1': [], 'r2': [], 'iter': []}

    def init_q(self, shift1, shift2):
        Q = {}
        agent1_possible_states = self.get_all_possible_states(shift=shift1)
        agent2_possible_states = self.get_all_possible_states(shift=shift2)
        for agent1 in agent1_possible_states:
            agent1_allowed_actions = self.get_allowed_actions(agent1)
            Q[agent1] = {}
            for agent2 in agent2_possible_states:
                temp = {}
                for action in agent1_allowed_actions:
                    temp[action()] = self.INIT_Q_VALUE
                Q[agent1][agent2] = temp
        return Q

    def get_possible_states_of_partner(self, agent, limit=3):
        possible_states = []
        for i in xrange(-limit + agent.x, limit + 1 + agent.x):
            for j in xrange(-limit + agent.y, limit + 1 + agent.y):
                possible_state = State(i, j, shift=agent.shift)
                if self.is_it_possible_position(possible_state):
                    possible_states.append(possible_state)
        return possible_states

    def get_all_possible_states(self, shift):
        possible_states = []
        for i in xrange(self.WIDTH):
            for j in xrange(self.HEIGHT):
                possible_states.append(State(i, j, shift=shift))
        return possible_states

    def is_it_possible_position(self, state):
        return 0 <= state.x < self.WIDTH and 0 <= state.y < self.HEIGHT

    def is_movable_to_the_left(self, agent):
        return agent.x - agent.shift > 0

    def is_movable_to_the_right(self, agent):
        return agent.x + agent.shift < self.WIDTH

    def is_movable_to_the_top(self, agent):
        return agent.y - agent.shift > 0

    def is_movable_to_the_bottom(self, agent):
        return agent.y + agent.shift < self.HEIGHT

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

    def choose_next_action(self, agent_a, agent_b, agent_a_Q, randomly=True):
        allowed = list(agent_a_Q[agent_a][agent_b].items())
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
        allowed = list(Q[st1][st2].items())
        max_v = allowed[0][1]
        for allowed_state in allowed:
            if allowed_state[1] > max_v:
                max_v = allowed_state[1]
        return max_v

    def get_updated_q(self, Q, st1, st2, action, alpha, r, maxQ):
        """
            Q[s',a'] = Q[s',a'] + alpha * (reward + gamma * MAX(Q,s) - Q[s',a'])
            #  s' -> old state
        """
        return Q[st1][st2][action] + alpha * (r + self.GAMMA * maxQ - Q[st1][st2][action])

    def is_game_failed(self, state):
        return self.ROOM[state.y][state.x] == self.DEATH

    def is_agents_too_far_away(self, st1, st2):
        return self.calc_distance(st1, st2) > 3

    def calc_distance(self, st1, st2):
        return (pow(st1.x - st2.x, 2) + pow(st1.y - st2.y, 2)) ** .5

    def is_game_won(self, state):
        return self.ROOM[state.y][state.x] == self.WIN

    def get_rewards(self, act1, act2, st1, st2):
        r1 = r2 = self.WALK_REWARDS
        restart = False
        fail = False
        win = False

        if act1.x == act2.x and act1.y == act2.y:
            # r1 = r2 = -1.
            pass

        if self.is_game_won(act1) or self.is_game_won(act2):
            r1 = r2 = self.WIN
            win = True

        if self.is_agents_too_far_away(act1, st2) and self.is_agents_too_far_away(act1, act2):
            r1 = self.DEATH
            fail = True

        if self.is_agents_too_far_away(act2, st1) and self.is_agents_too_far_away(act1, act2):
            r2 = self.DEATH
            fail = True

        if self.is_game_failed(act1):
            r1 = self.DEATH
            fail = True

        if self.is_game_failed(act2):
            r2 = self.DEATH
            fail = True

        if fail:
            self.failures += 1
            restart = True
        elif win:
            self.success += 1
            restart = True

        return r1, r2, restart

    def mark_worst_as_dangerous(self, q, st1, st2, action, rewards):
        if rewards == self.DEATH:
            self.mark_danger_states(q, st1, st2, action)

    def next_actions(self, st1, st2):
        return self.choose_next_action(st1, st2, self.q1), self.choose_next_action(st2, st1, self.q2)

    @timer
    def training(self, start_st1, start_st2):
        alpha = 1
        st1, st2 = start_st1, start_st2
        act1, act2 = self.next_actions(st1, st2)
        self.FRAME_RATE = 0.1

        while self.failures < self.MAX_ITERATIONS and self.success < self.EPOCHS:

            old_st1, old_st2 = st1, st2
            old_act1, old_act2 = act1, act2

            r1, r2, restart = self.get_rewards(act1, act2, st1, st2)

            max_q1 = self.get_max_q(self.q1, act1, act2)
            max_q2 = self.get_max_q(self.q2, act2, act1)

            u1 = self.get_updated_q(self.q1, old_st1, old_st2, old_act1, alpha, r1, max_q1)
            u2 = self.get_updated_q(self.q2, old_st2, old_st1, old_act2, alpha, r2, max_q2)

            self.q1[old_st1][old_st2][old_act1] = u1
            self.q2[old_st2][old_st1][old_act2] = u2

            self.mark_worst_as_dangerous(self.q1, st1, st2, act1, r1)
            self.mark_worst_as_dangerous(self.q2, st2, st1, act2, r2)

            if restart:
                st1, st2 = start_st1, start_st2
            else:
                st1, st2 = act1, act2

            act1, act2 = self.next_actions(st1, st2)

            alpha = pow(self.tick, -0.1)
            self.tick += 1

            self.show_statistics()

            self.capture_statistics(r1, r2, u1, u2)

    def capture_statistics(self, r1, r2, u1, u2):
        self.statistics["Q1"].append(u1)
        self.statistics["Q2"].append(u2)
        self.statistics["r1"].append(r1)
        self.statistics["r2"].append(r2)
        self.statistics["iter"].append(self.tick)

    def mark_danger_states(self, q, st1, st2, act):
        del (q[st1][st2][act])

    def show_progress(self, st1, st2, st1_action, st2_action, q1, q2):
        os.system('cls' if os.name == 'nt' else 'clear')
        for i in xrange(self.HEIGHT):
            for j in xrange(self.WIDTH):
                if st1_action.x == st2_action.x == j and st1_action.y == st2_action.y == i:
                    print('\033[0;35;40m%6.2f' % q1[st1][st2][st1_action], end='')
                elif st1_action.x == j and st1_action.y == i:
                    print('\033[0;34;40m%6.2f' % q1[st1][st2][st1_action], end='')
                elif st2_action.x == j and st2_action.y == i:
                    print('\033[0;33;40m%6.2f' % q2[st2][st1][st2_action], end='')
                else:
                    if self.ROOM[i][j] == self.DEATH:
                        print('\033[1;31;40m%6s' % "x", end='')
                    elif self.ROOM[i][j] == self.WIN:
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
        self.FRAME_RATE = 0.35
        next_st1, next_st2 = st1, st2
        print("Final result")
        while True:

            if self.is_game_won(next_st1) or self.is_game_won(next_st2):
                print("Steps: %d" % steps)
                break

            next_st1 = self.choose_next_action(st1, st2, q1, False)
            next_st2 = self.choose_next_action(st2, st1, q2, False)

            self.show_progress(st1, st2, next_st1, next_st2, self.q1, self.q2)

            st1 = next_st1
            st2 = next_st2

            steps += 1

    def show_statistics(self):
        sys.stdout.write('\rSuccess: %d, Failures: %d' % (self.success, self.failures))
        sys.stdout.flush()

    # def show_graph(self):
    #     plt.subplot(221)
    #     plt.plot(self.statistics["iter"], self.statistics["Q1"], lw=1)
    #     plt.title('Iter/Q1')
    #
    #     plt.subplot(222)
    #     plt.plot(self.statistics["iter"], self.statistics["Q2"], lw=1)
    #     plt.title('Iter/Q2')
    #
    #     plt.subplot(223)
    #     plt.plot(self.statistics["iter"], self.statistics["r1"], lw=1)
    #     plt.title('Iter/Rewards1')
    #
    #     plt.subplot(224)
    #     plt.plot(self.statistics["iter"], self.statistics["r2"], lw=1)
    #     plt.title('Iter/Rewards2')
    #     plt.show()


def run():
    q_learn = QLearning()

    agent1 = State(1, 8, shift=1)
    agent2 = State(0, 8, shift=2)

    q_learn.q1 = q_learn.init_q(agent1.shift, agent2.shift)
    q_learn.q2 = q_learn.init_q(agent2.shift, agent1.shift)

    q_learn.training(agent1, agent2)

    q_learn.show_final_result(agent1, agent2, q_learn.q1, q_learn.q2)

    # q_learn.show_graph()


if __name__ == '__main__':
    run()
