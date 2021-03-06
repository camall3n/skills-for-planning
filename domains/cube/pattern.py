import glob
import random
from domains.cube import formula

CUBE_IN_CUBE = "F L F U' R U F F L L U' L' B D' B' L L U".split()
EXCHANGED_RINGS = "B' U' B' L' D B U D D B U L D' L' U' L L D".split()
TWISTED_PEAKS = "F B' U F U F U L B L L B' U F' L U L' B".split()
EXCHANGED_PEAKS = "F U U L F L' B L U B' R' L' U R' D' F' B R R".split()
TWISTED_CROSSES = "R R L' D F F R' D' R' L U' D R D B B R' U D D".split()
SIX_SPOTS = "U D' R L' F B' U D'".split()
SUPERFLIP_QTM = "U U F U U R' L F F U F' B' R L U U R U D' R L' D R' L' D D".split()

SCRAMBLE_1 = ("U D' R F R R' R' R' F' L' D R R' L L' L' D' L B' R' F B' D D' R B L L L F' U' "
              "L L' F' D L' B' L U' D R' R' U' D B D F' D R' F L L' U' F' R U F' B' F R").split()


def scramble(seed=0, length=60):
    """Generate the canonical scramble for a particular random seed"""
    old_st = random.getstate()
    random.seed(seed)
    formula_ = formula.random_formula(length)
    random.setstate(old_st)
    return formula_

def buchner2018pattern(seed=0):
    filenames = glob.glob('domains/cube/buchner2018/scrambles/id-{:03d}_*.txt'.format(seed))
    assert len(filenames)==1
    with open(filenames[0], 'r') as file:
        pattern = file.read()
        formula_ = formula.simplify(pattern.split())
    return formula_

def test():
    """Test scramble patterns"""
    assert scramble(1) == scramble(1)
    assert scramble(1) != scramble(2)
    assert scramble(1) == SCRAMBLE_1
    print('All tests passed.')

if __name__ == '__main__':
    test()
