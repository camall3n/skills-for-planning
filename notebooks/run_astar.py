import argparse
import copy
import pickle
import os
import sys
from cube import cube
from notebooks import astar
from cube import options
from cube import pattern
from matplotlib import pyplot as plt

if 'ipykernel' in sys.argv[0]:
    sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser()
parser.add_argument('--scramble_seed','-s', type=int, default=1,
                    help='Seed to use for initial scramble')
parser.add_argument('--skill_mode','-m', type=str, default='expert',
                    choices=['primitive','expert','fixed_random','full_random','generated'],
                    help='Type of skills to consider during search')
parser.add_argument('--random-goal','-r', action='store_true', default=False,
                    help='Generate a random goal instead of the default solve configuration')
parser.add_argument('--skill_version','-v', type=str, default='0.2',
                    choices=['0.1','0.2','0.3'],
                    help='Which version to use for generated skills')
parser.add_argument('--max_transitions', type=lambda x: int(float(x)), default=1e5,
                    help='Maximum number of variables changed per primitive action')
args = parser.parse_args()

seed = args.scramble_seed
skill_mode = args.skill_mode
cost_mode = 'per-skill'
debug = False

# Set up the scramble
newcube = cube.Cube()
scramble = pattern.scramble(seed)

start = copy.deepcopy(newcube)
start.apply(scramble)
print('Using scramble: {:03d}'.format(seed))
start.render()

# Define the skills
if skill_mode == 'primitive':
    skills = options.primitive.actions
    models = options.primitive.models
elif skill_mode == 'expert':
    skills = options.primitive.actions + options.expert.options
    models = options.primitive.models + options.expert.models
elif 'random' in skill_mode:
    if skill_mode == 'full_random':
        options.set_random_skill_seed(seed)
    skills = options.primitive.actions + options.random.options
    models = options.primitive.models + options.random.models
elif skill_mode == 'generated':
    options.load_generated_skills(args.skill_version)
    skills = options.primitive.actions + options.generated.options
    models = options.primitive.models + options.generated.models

if args.random_goal:
    goal = cube.Cube().apply(pattern.scramble(seed+1000))
    print('Using goal pattern: {:03d}'.format(seed+1000))
else:
    goal = newcube

# Set up the search problem
is_goal = lambda node: node.state == goal
heuristic = lambda cube: len(cube.summarize_effects(baseline=goal))

if cost_mode == 'per-action':
    step_cost = lambda skill: len(skill)
elif cost_mode == 'per-skill':
    step_cost = lambda skill: 1

def get_successors(cube):
    return [(copy.deepcopy(cube).apply(swap_list=m), s) for s,m in zip(skills, models)]

#%% Run the search
search_results = astar.search(start, is_goal, step_cost, heuristic, get_successors, args.max_transitions)

#%% Save the results
tag = skill_mode
if skill_mode == 'generated':
    tag += '-v{}'.format(args.skill_version)
if args.random_goal:
    tag = 'random_goal/'+tag
else:
    tag = 'default_goal/'+tag
results_dir = 'results/planning/{}/'.format(tag)
os.makedirs(results_dir, exist_ok=True)
with open(results_dir+'/seed-{:03d}.pickle'.format(seed), 'wb') as f:
    pickle.dump(search_results, f)
