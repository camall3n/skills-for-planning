import random as pyrandom
from cube import cube, formula, skills

class primitive:
    alg_formulas = [[a] for a in cube.actions]
    actions = alg_formulas
    models = [cube.Cube().apply(a).summarize_effects() for a in actions]

class expert:
    alg_formulas = [
        skills.swap_3_edges_face,
        skills.swap_3_edges_mid,
        skills.swap_3_corners,
        skills.orient_2_edges,
        skills.orient_2_corners,
    ]
    options = [variation for f in alg_formulas for variation in formula.variations(f)]
    models = [cube.Cube().apply(o).summarize_effects() for o in options]

def set_random_skill_seed(seed):
    st = pyrandom.getstate()
    pyrandom.seed(seed)
    formulas = [skills.random_skill(len(a)) for a in expert.alg_formulas]
    pyrandom.setstate(st)

    class uniform_random:
        random_seed = seed
        alg_formulas = formulas
        options = [variation for f in alg_formulas for variation in formula.variations(f)]
        models = [cube.Cube().apply(o).summarize_effects() for o in options]
    global random
    random = uniform_random

set_random_skill_seed(0)

# class conjugates:
#     alg_formulas = [skills.random_conjugate(len(a)) for a in expert.alg_formulas]
#     options = [variation for f in alg_formulas for variation in formula.variations(f)]
#     models = [cube.Cube().apply(o).summarize_effects() for o in options]
#
# class commutators:
#     alg_formulas = [skills.random_commutator(len(a)) for a in expert.alg_formulas]
#     options = [variation for f in alg_formulas for variation in formula.variations(f)]
#     models = [cube.Cube().apply(o).summarize_effects() for o in options]