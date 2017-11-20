#!/usr/bin/env python
from __future__ import print_function
import random

# def view():
#     import time
#     import curses
#
#     def pbar(window):
#         for i in range(10):
#             window.addstr(10, 10, "[" + ("=" * i) + ">" + (" " * (10 - i)) + "]")
#             window.refresh()
#             time.sleep(0.5)
#
#     curses.wrapper(pbar)

class QLearning():
    def __init__(self):
        self.WIDTH = 16
        self.HEIGHT = 9

        self.Q = [[0 for x in range(self.WIDTH)] for y in range(self.HEIGHT)]

        # Rewards matrix
        self.R = [
            [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -100, -1, -1],
            [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -100, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
            [-1, -1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1, 100]
        ]

        self.Gamma = .8
        self.Epochs = 1000

    def next_state(self):
        pass


def run():
    pass


if __name__ == '__main__':
    run()
