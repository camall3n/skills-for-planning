from collections import namedtuple

import numpy as np

NAME = 'npuzzle'
PLOTS = ['learning_curves', 'planning_boxes']
SUMMARIES = ['macro_type', 'goal_type']
FIELDS = 'alg', 'puzzle_size', 'goal_type', 'macro_type', 'seed'
FIGSIZE = (8,6)
FONTSIZE = 18
HLINE = None
TRANSITION_CAP = 5e5
XLIM = [-100,TRANSITION_CAP]
YLIM = [0,16]

PlotVars = namedtuple('PlotVars', ['macro_type', 'goal_type', 'color', 'zorder'])
PLOT_VARS = [
    PlotVars('random', 'default_goal', 'C2', 5),
    PlotVars('primitive', 'default_goal', 'C0', 10),
    PlotVars('learned', 'default_goal', 'C3', 10),
]


def get_goal(state, metadata):
    if metadata.goal_type == 'default_goal':
        goal = state.reset()
    else:
        goal = state.reset().scramble(seed=metadata.seed+1000)
    return goal

def heuristic(state, goal):
    return len(state.summarize_effects(baseline=goal)[0])

def get_primitive_steps(sequence):
    return len(np.concatenate(sequence))

def get_macro_steps(sequence):
    return len(sequence)

def get_macro_lengths(sequence):
    return list(map(len, sequence))