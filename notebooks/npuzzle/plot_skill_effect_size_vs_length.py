import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from tqdm import tqdm

from npuzzle import npuzzle
from npuzzle import options

filter = lambda x,y: zip(*[(x,y) for x,y in zip(x,y) if x!=1])
rnd_skill_len = list(map(len,options.random.options[(0,0)]))
rnd_skill_size = list(map(lambda x: len(x[0]),options.random.models[(0,0)]))
rnd_skill_len, rnd_skill_size = filter(rnd_skill_len, rnd_skill_size)

gen_skill_len = list(map(len,options.generated.options[(0,0)]))
gen_skill_size = list(map(lambda x: len(x[0]),options.generated.models[(0,0)]))
gen_skill_len, gen_skill_size = filter(gen_skill_len, gen_skill_size)

#%%
noise = 0.
offset = 0.1
fig, ax = plt.subplots(figsize=(8,6))
# plt.grid('on')
x = [1-offset]
y = [2]
plt.scatter(x,y, c='C0', s=70, marker='o', label='primitive actions')
x = np.asarray(rnd_skill_len)+offset+np.random.normal(0,noise,len(rnd_skill_len))
y = np.asarray(rnd_skill_size)+np.random.normal(0,noise,len(rnd_skill_size))
plt.scatter(x,y, c='C2', s=70, marker='^', label='random skills')
x = np.asarray(gen_skill_len)+np.random.normal(0,noise,len(gen_skill_len))
y = np.asarray(gen_skill_size)+np.random.normal(0,noise,len(gen_skill_size))
plt.scatter(x,y, c='C3', s=70, marker='+', label='generated skill')
plt.xlabel('Number of primitive actions per skill')
plt.ylabel('Number of variables modified')
plt.xlim([0,20])
plt.ylim([0,12])
plt.xticks(range(21))
plt.gca().set_axisbelow(True)
plt.legend(loc='upper left')
plt.title('Skill entanglement by type (15-puzzle)')
plt.savefig('results/plots/npuzzle/npuzzle_skill_entanglement_by_type.png')
plt.show()

#%% Visualize some options
for blank_idx in [(3,3)]:#options.generated.models.keys():
    option_list = options.generated.options[blank_idx]
    model_list = options.generated.models[blank_idx]
    for i in range(len(option_list)):
        option = option_list[i]
        model = model_list[i]
        if len(model[0]) == 2 and len(option) == 19:
            puz = npuzzle.NPuzzle(n=15, start_blank=blank_idx)
            print(puz)
            puz.apply_macro(model=model)
            print(option)
            print(model)
            print(puz)
            print()