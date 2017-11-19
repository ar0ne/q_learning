#!/usr/bin/env python

import numpy as np
import random

def view():
    import time
    import curses

    def pbar(window):
        for i in range(10):
            window.addstr(10, 10, "[" + ("=" * i) + ">" + (" " * (10 - i )) + "]")
            window.refresh()
            time.sleep(0.5)

    curses.wrapper(pbar)


# Rewards matrix
R =             [ [-1,-1,-1,-1, 0, -1],
                [-1,-1,-1, 0,-1,100],
                [-1,-1,-1, 0,-1, -1],
                [-1, 0, 0,-1 ,0, -1],
                [-1, 0, 0,-1,-1,100],
                [-1, 0,-1,-1, 0,100] ]

Q =          [ [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0] ]
Gamma = .8

MaxStateCount = 6

# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]

GoalState = 5


def get_allowed_actions(state):
    global R
    allowed_actions = []
    for i in xrange(MaxStateCount):
        if R[state][i] >= 0:
            allowed_actions.append(i)
    return allowed_actions


def choose_next_action(current_state):
    allowed = get_allowed_actions(current_state)
    return random.choice(allowed)

def get_max_Q(next_state):
    global Q
    allowed = get_allowed_actions(next_state)
    max = 0
    for i in allowed:
        if Q[next_state][i] > max:
            max = Q[next_state][i]
    return max


def calculate_Q(state, action):
    global R, Q, Gamma
    return R[state][action] + Gamma * get_max_Q(action)


def run():
    global Q, GoalState, MaxStateCount
    # init state    
    state = random.choice(range(MaxStateCount))  
    count = 0
    while count < 1000:
        
        action = choose_next_action(state)

        Q[state][action] = calculate_Q(state, action)

        state = action

        if(state == GoalState):
            print("Finished")
            count += 1
            state = random.choice(range(MaxStateCount))

    print(Q)

if __name__ == '__main__':
    run()
