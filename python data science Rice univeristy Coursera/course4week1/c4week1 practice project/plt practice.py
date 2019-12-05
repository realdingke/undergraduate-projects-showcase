import matplotlib.pyplot as plt
height_data=[173,172,170,178,179,190,180,185,183,171,188,186,175,176,177,179,191]
height_groups=[170,172,174,176,178,180,182,184,186,188,190,192]
plt.hist(height_data,height_groups,color='r',histtype='bar',rwidth=0.9)
plt.plot(height_data,[x for x in range(len(height_data))])
plt.xlabel('x')
plt.ylabel('y')
plt.title('practice plots\nnice try')
plt.show()