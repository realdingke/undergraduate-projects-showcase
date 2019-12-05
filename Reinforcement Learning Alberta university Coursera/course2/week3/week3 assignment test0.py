import numpy as np
policy = np.ones(shape=(12*4, 4)) * 0.25
### START CODE HERE ###
for i in [36,24,12]:
    policy[i]=[1,0,0,0]
for i in [11,23,35]:
    policy[i]=[0,0,1,0]
for i in range(0,11):
    policy[i]=[0,0,0,1]
for idx,actions in enumerate(policy):
    if (actions==np.array([0.25,0.25,0.25,0.25])).all():
        policy[idx]=[1,0,0,0]
print(policy)