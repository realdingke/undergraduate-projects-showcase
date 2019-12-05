import numpy as np
A=[0,1,2]
V=np.zeros(4)
s=0
trans=      [[[0,0.24],
                [1,0.22],
                [2,0.18],
                [1,0.36]],
            [[0,0.24],
                [1,0.22],
                [2,0.18],
                [1,0.36]],
            [[0,0.25],
                [1,0.21],
                [2,0.18],
                [1,0.36]]]
transition=np.array(trans)
pi=np.array([0.75,0.21,0.04])
for act in A:
        for s_, (r,p) in enumerate(transition[act]):
            V[s]+=pi[act]*(p)*(r+0.9*(V[s_]))
print(V)
print(np.zeros(3))