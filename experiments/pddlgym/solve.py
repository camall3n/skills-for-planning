import argparse
import copy
import os
import pickle
import random
import sys
from types import SimpleNamespace

import gym
import pddlgym
from pddlgym.structs import LiteralConjunction

# from domains.npuzzle import NPuzzle, macros
from experiments import search, iw, bfws
from domains.pddlgym.macros import load_learned_macros
from domains.pddlgym.pddlgymenv import scramble

def parse_args():
    """Parse input arguments

    Use --help to see a pretty description of the arguments
    """
    if 'ipykernel' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hanoi',
                        help='Name of PDDL domain')
    parser.add_argument('--seed','-s', type=int, default=0,
                        help='Seed to use for RNGs and problem index')
    parser.add_argument('--macro_type','-m', type=str, default='primitive',
                        choices=['primitive', 'learned'],
                        help='Type of macro_list to consider during search')
    parser.add_argument('--search_alg', type=str, default='gbfs',
                        choices = ['astar', 'gbfs', 'weighted_astar', 'bfws_r0', 'bfws_rg'],
                        help='Search algorithm to run')
    parser.add_argument('--g_weight', type=float, default=None,
                        help='Weight for g-score in weighted A*')
    parser.add_argument('--h_weight', type=float, default=None,
                        help='Weight for h-score in weighted A*')
    parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=1e5,
                        help='Maximum number of state transitions')
    parser.add_argument('--bfws_precision', type=int, default=3,
                        help='The number of width values, w \in {1,...,P}, to use when the search algorithm is best-first width search')
    return parser.parse_args()

def solve():
    """Instantiate PDDL domain with PDDLGym and solve with the specified macro-actions and search algorithm"""
    args = parse_args()

    # Set up the domain
    env = gym.make("PDDLEnv-Gen-{}-v0".format(args.env_name.capitalize()))
    env.fix_problem_index(args.seed)
    # start = scramble(env, seed=args.seed)
    random.seed(args.seed)
    if args.macro_type == 'learned':
        env = load_learned_macros(env, 0)
        env.fix_problem_index(args.seed)
    env._render = None
    env.seed(args.seed)

    start, _ = env.reset()
    env.set_state(start)
    goal = start.goal
    assert isinstance(goal, LiteralConjunction)
    heuristic = lambda obs: len([lit for lit in goal.literals if lit not in obs.literals])

    print('Using seed: {:03d}'.format(args.seed))
    print('Objects:', sorted(list(start.objects)))
    print('Goal:', goal)

    # Set up the search problem
    search_fn = {
        'astar': search.astar,
        'gbfs': search.gbfs,
        'weighted_astar': search.weighted_astar,
        'bfws_r0': bfws.bfws,
        'bfws_rg': bfws.bfws,
    }[args.search_alg]

    def restore_state(state):
        env.set_state(state)
        return env

    def get_successors(state):
        env.set_state(state)
        valid_actions = sorted(list(env.action_space.all_ground_literals(state)))
        random.shuffle(valid_actions)
        successors = [(restore_state(state).step(a)[0], [a]) for a in valid_actions]
        return successors

    heuristic = lambda state: len([lit for lit in state.goal.literals if lit not in state.literals])

    search_dict = {
        'start': start,
        'is_goal': lambda node: heuristic(node.state) == 0,
        'step_cost': lambda macro: 1,
        'heuristic': heuristic,
        'get_successors': get_successors,
        'max_transitions': args.max_transitions,
    }

    if args.search_alg == 'weighted_astar':
        assert (args.g_weight is not None
                and args.h_weight is not None), 'Must specify weights if using weighted A*.'
        gh_weights = (args.g_weight, args.h_weight)
        search_dict['gh_weights'] = gh_weights

    if 'bfws' in args.search_alg:
        search_dict['precision'] = args.bfws_precision
    if args.search_alg == 'bfws_rg':
        goal_fns = [(lambda x, i=i: goal.literals[i] in x.state.literals) for i, _ in enumerate(goal.literals)]
        relevant_atoms = iw.iw(1, start, get_successors, goal_fns)
        if not relevant_atoms:
            relevant_atoms = iw.iw(2, start, get_successors, goal_fns)
        if not relevant_atoms:
            relevant_atoms = None
        search_dict['R'] = relevant_atoms

    #%% Run the search
    search_results = search_fn(**search_dict)

    #%% Save the results
    tag = '{}'.format(args.env_name)
    tag += '/{}'.format(args.macro_type)

    results_dir = 'results/pddlgym-gen/{}/{}/'.format(args.search_alg,tag)
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir+'seed-{:03d}.pickle'.format(args.seed), 'wb') as file:
        pickle.dump(search_results, file)

    plan = search_results[1]
    print("Plan length:", len(plan))
    env.close()

if __name__ == '__main__':
    solve()
