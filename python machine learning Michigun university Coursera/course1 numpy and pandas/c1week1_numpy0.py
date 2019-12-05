import numpy as np
#a1=np.array([1,2,3]*2)
#a2=a1.reshape(3,2)
##a2=a1.resize(3,2) This will return none, as resize only changes a1, does not create a2
#print(a1,'\n')
#print(a2)
#print(a1*2)
#a3=np.repeat([3,4,5,6],2)
#print(a3**2)
#a4=np.arange(10)**2
#print(a4)
#print(a4[-3::-1])  #slice from -3 position backwards with an increment of 1
#print(a4[a4<50])   #conditional slicing
#a4[a4>=50]=50
#print(a4)          #assign 50 to elements greater than 49
#a5=np.random.randint(0,10,(4,3))
#print(a5)
#print(a5.reshape(a5.shape[0]*a5.shape[1])[::4]) #reshape to a single line array and get element with increment of 4
#for x,y in zip(a5[0,:-1],(a5**2)[1,:-1]):   #zip takes in mutiple of iterables and return tuples with elements of each instance in each iterable
#    print(x,y)
#print(list(zip(a5,a5**2)))
#a=np.array([[2]*8]*8).reshape(8,8)
#b=np.array([[1]*8]*8).reshape(8,8)
#c=np.hstack((a,b))
#print(np.dot(a,b))
#A=np.array([[1,2,3,4]]*3)
#B=A.sum(axis=1,keepdims=True)
#C=np.exp(A)
#print(np.array([[1,2],[2,4]]))
print(np.random.rand(1,1))