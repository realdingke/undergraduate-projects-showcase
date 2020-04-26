import ast
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import math
def read_csv_file(file_name, file_delimeter):     
    with open(file_name, newline='') as csv_file:       # don't need to explicitly close the file now
        csv_table = []
        csv_reader = csv.reader(csv_file, delimiter=file_delimeter)
        idx=0
        for row in csv_reader:
            if idx==0 or idx==1 or idx==2 or idx==4 or idx==5 or idx==6:
                csv_table.append([ast.literal_eval(i) for i in row])
            else:
                csv_table.append(['']+[float(val) for val in row[1:]])
            idx+=1
    return csv_table

#set the x and y range for the map, establish the 2-D cartesian space for waterway
X_RANGE=100
Y_RANGE=100
#set the width and length for the ship itself
W=3
L=8
#set the starting and winning state, with x,y coodinates and heading angle
START=(20,10,0)
#WIN=(90,90,0)
WIN=[]
for i in range(88,93):
    for j in range(88,93):
        for k in range(0,360,10):
            WIN.append((i,j,k))
WIN_AVR=(90,90,0)

#read all the data from csv
final_xy_0=read_csv_file('csv_single_obs_data',',')[0]
final_0=read_csv_file('csv_single_obs_data',',')[1]
sum_reward_0=read_csv_file('csv_single_obs_data',',')[3]
final_xy_1=read_csv_file('csv_single_obs_data',',')[4]
final_1=read_csv_file('csv_single_obs_data',',')[5]
sum_reward_1=read_csv_file('csv_single_obs_data',',')[7]

#plot the data on the same graph
fig0=plt.figure()
plt.axes()
fig0.gca().set_facecolor('xkcd:sky blue')
plt.scatter(START[0],START[1],marker='o',c='w',s=5)
plt.scatter(WIN_AVR[0],WIN_AVR[1],marker='o',c='g',s=50)
for idx,val in {(10,25):(50,15)}.items():
    rect=plt.Rectangle(idx,val[0],val[1],fc='r')
    fig0.gca().add_patch(rect)
fig0.gca().add_patch(plt.Polygon(final_xy_0, closed=None, fill=None, edgecolor='y', label='with constant reward'))
fig0.gca().add_patch(plt.Polygon(final_xy_1, closed=None, fill=None, edgecolor='k', label='with L reward function'))
plt.legend()
ship_s=mp.Ellipse(START[:-1],W,L,angle=-START[2],fill=False,ec='w')
fig0.gca().add_patch(ship_s)
# for x,y,ang in final_0[::5]:
#     ship_m=mp.Ellipse((x,y),W,L,angle=-ang,fill=False,ec='xkcd:purple')
#     fig0.gca().add_patch(ship_m)
# for x1,y1,ang1 in final_1[::5]:
#     ship_m1=mp.Ellipse((x1,y1),W,L,angle=-ang1,fill=False,ec='xkcd:black')
#     fig0.gca().add_patch(ship_m1)
plt.axis('scaled')
plt.show()


#plot the reward curve
fig1=plt.figure()
plt.plot(list(range(0,len(sum_reward_0[1000:]),10)),sum_reward_0[1000::10],c='b',label='with constant reward')
plt.plot(list(range(0,len(sum_reward_1[1000:]),10)),sum_reward_1[1000::10],c='y',label='with L reward function')
plt.legend()
plt.xlabel('episode number')
plt.ylabel('sum of rewards per episode')
plt.show()

print(len(read_csv_file('csv_single_obs_data',',')[2]),len(read_csv_file('csv_single_obs_data',',')[6]))
#print(sum_reward_0[::10])