import numpy as np
def argmax(q_values):
    top = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        if q_values[i]>top:
            top=q_values[i]
            ties=[]
        if q_values[i]==top:
            ties.append(i)
        
    return np.random.choice(ties)

print(argmax([0,0,0,0,0,0,0,1,0]))
print(np.random.choice([i for i in range(len([1,2,3,4,5]))]))
print(np.random.random())
