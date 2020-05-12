import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import math
import csv
import ast
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

#define a random obstacle generation
def obs(number=3):
    """randomly generate up to n obstacles with randomly generated width and height,
       within the map area
    """
    obs_dict={}
    for num in range(np.random.choice([n for n in range(1,number)])):
        i=np.random.choice([n for n in range(80)])
        j=np.random.choice([n for n in range(36,69)])
        if (i,j) not in obs_dict.keys():
            obs_dict[(i,j)]=(np.random.choice([n for n in range(5,20)]),np.random.choice([n for n in range(5,20)]))
    return obs_dict
OBS=obs(4)

#the environment to define ship motion:
class Env():
    def __init__(self,state=START):
        #self.obstacles=OBS
        self.obstacles={(10,25):(50,15)}           #can overwrite manually
        #self.obstacles={(10,25):(50,10),(50,60):(50,10)}
        self.state=state
        self.End=False
        #self.time=0
        self.step=15
        self.V=0.19   #velocity for ship is assumed to be constant
        self.K=0.08    #defines the turning capability coefficient
        self.T=10.8    #defines the turning lag coefficient
#         reward=None
#         self.reward_state_termination=(reward,self.state,self.End)
        
    def obs_detect(self,state):
        """determines if the input state hits any obstacle, returns a boolean"""
        boolean=False
        for idx,val in self.obstacles.items():
            if state[0]>idx[0]-W/2 and state[0]<idx[0]+val[0]+W/2:
                if state[1]>idx[1]-L/2 and state[1]<idx[1]+val[1]+L/2:
                    boolean=True
        return boolean
    
    def bound_detect(self,state):
        """determine if the input is out of bound"""
        boolean=False
        if (state[0]-W/2)<0 or (state[0]+W/2)>X_RANGE:
            boolean=True
        if (state[1]-L/2)<0 or (state[1]+L/2)>Y_RANGE:
            boolean=True
        return boolean
    
    def distance_reward(self,state):
        tar_dist=math.sqrt((state[0]-WIN_AVR[0])**2+(state[1]-WIN_AVR[1])**2)
        return -0.03*tar_dist

    def nomoto_model(self,rudder_ang):
        """input a rudder angle, through the nomoto input-output model,
           returns the output ship course"
        """
        course=int(round(self.K*int(rudder_ang)*(self.step-self.T+self.T*np.exp(-self.step/self.T)),-1))
        return course

    def env_step(self,action):
        """defines the agent-environment interaction"""
        #total of five rudder angle actions:0,15,35,-15,-35
        if action=='0':
            pos_state_2=(self.state[2]+self.nomoto_model(action))%360
            pos_state_1=self.state[1]+self.V*np.cos(np.deg2rad(pos_state_2))*self.step
            pos_state_0=self.state[0]+self.V*np.sin(np.deg2rad(pos_state_2))*self.step
            self.state=(int(round(pos_state_0)),int(round(pos_state_1)),pos_state_2)
        elif action=='15':
            pos_state_2=(self.state[2]+self.nomoto_model(action))%360
            pos_state_1=self.state[1]+self.V*np.cos(np.deg2rad(pos_state_2))*self.step
            pos_state_0=self.state[0]+self.V*np.sin(np.deg2rad(pos_state_2))*self.step
            self.state=(int(round(pos_state_0)),int(round(pos_state_1)),pos_state_2)
        elif action=='35':
            pos_state_2=(self.state[2]+self.nomoto_model(action))%360
            pos_state_1=self.state[1]+self.V*np.cos(np.deg2rad(pos_state_2))*self.step
            pos_state_0=self.state[0]+self.V*np.sin(np.deg2rad(pos_state_2))*self.step
            self.state=(int(round(pos_state_0)),int(round(pos_state_1)),pos_state_2)
        elif action=='-15':
            pos_state_2=(self.state[2]+self.nomoto_model(action))%360
            pos_state_1=self.state[1]+self.V*np.cos(np.deg2rad(pos_state_2))*self.step
            pos_state_0=self.state[0]+self.V*np.sin(np.deg2rad(pos_state_2))*self.step
            self.state=(int(round(pos_state_0)),int(round(pos_state_1)),pos_state_2)
        elif action=='-35':
            pos_state_2=(self.state[2]+self.nomoto_model(action))%360
            pos_state_1=self.state[1]+self.V*np.cos(np.deg2rad(pos_state_2))*self.step
            pos_state_0=self.state[0]+self.V*np.sin(np.deg2rad(pos_state_2))*self.step
            self.state=(int(round(pos_state_0)),int(round(pos_state_1)),pos_state_2)
        else:
            raise Exception('action is not recognised')
        
        
        #reward=-1
        self.End=False
        #set reward and update state for each case
        if self.state in WIN:   #terminate the episode
            reward=1000
            self.End=True
        elif self.obs_detect(state=self.state):    #hits the obstacles
            reward=-1000
            self.End=False
            self.state=START
        elif self.bound_detect(state=self.state):   #out of bound
            reward=-100
#             reward=-1000
            self.End=False
            self.state=START
        else:                           #negative reward for any other steps to speed up learning
            #reward=-1
            reward=self.distance_reward(self.state)        #use distance reward function
            self.End=False
            
        self.reward_state_termination=(reward,self.state,self.End)
        return self.reward_state_termination
        
        
    
    def env_reset(self):
        #self.time=0
        self.state=START
        self.End=False
    
    
    def show_grid(self,final=[],final_xy=[],show_ship=False):
        fig0=plt.figure()
        plt.axes()
        fig0.gca().set_facecolor('xkcd:sky blue')
        plt.scatter(START[0],START[1],marker='o',c='w',s=5)
        plt.scatter(WIN_AVR[0],WIN_AVR[1],marker='o',c='g',s=50)
        for idx,val in self.obstacles.items():
            rect=plt.Rectangle(idx,val[0],val[1],fc='r')
            fig0.gca().add_patch(rect)
        if len(final_xy)>0:
            fig0.gca().add_patch(plt.Polygon(final_xy, closed=None, fill=None, edgecolor='y'))
        if show_ship:
            ship_s=mp.Ellipse(START[:-1],W,L,angle=-START[2],fill=False,ec='xkcd:purple')
            fig0.gca().add_patch(ship_s)
            for x,y,ang in final[::5]:
                ship_m=mp.Ellipse((x,y),W,L,angle=-ang,fill=False,ec='xkcd:purple')
                fig0.gca().add_patch(ship_m)
        plt.axis('scaled')
        plt.show()

Env().show_grid(show_ship=True)



#define the q learning agent class
class Q_agent():
    def __init__(self):
        self.agent_Env=Env()
        self.actions=['0','15','35','-15','-35']
        self.agent_End=self.agent_Env.End
        self.step_size=0.2
        self.discount=0.9
        self.epsilon=0.1
        #populate the action value q(s,a) table with 0 first
        self.q_values={}
        for i in range(X_RANGE):
            for j in range(Y_RANGE):
                for k in range(0,360,10):
                    self.q_values[(i,j,k)]={}
                    for action in self.actions:
                        self.q_values[(i,j,k)][action]=0
        self.trace=[]     #keep a trace
        self.sum_reward_list=[None]
        
        
    
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values : the table of action-values
        Returns:
            action: an action with the highest value
        """
        top=float("-inf")
        ties=[]

        for i in self.actions:
            if q_values[i]>top:
                top=q_values[i]
                ties=[]

            if q_values[i]==top:
                ties.append(i)

        return np.random.choice(ties)
    
    def agent_start(self,state):
        """call to initiate the agent"""
        # Choose action using epsilon greedy.
        current_q=self.q_values[state]
        if np.random.uniform(0,1)<self.epsilon:
            action=np.random.choice(self.actions)
        else:
            action = self.argmax(current_q)
        self.prev_state=state
        self.prev_action=action
        return action
    
    
    def agent_step(self, reward, state):
        """Agent takes a step
        Input reward from last action and state determined by the agent-environment interaction
        Returns:the action the agent is taking.
        """
        # Choose action using epsilon greedy.
        current_q=self.q_values[state]
        if np.random.uniform(0,1)<self.epsilon:
            action=np.random.choice(self.actions)
        else:
            action = self.argmax(current_q)
        
        self.q_values[self.prev_state][self.prev_action]+=self.step_size*(reward+self.discount*self.q_values[state][action]-self.q_values[self.prev_state][self.prev_action])
        
        self.prev_state=state
        self.prev_action=action
        return action
        
    def agent_end(self,reward):
        self.q_values[self.prev_state][self.prev_action]+=self.step_size*(reward+0-self.q_values[self.prev_state][self.prev_action])
    
    def reset(self):
        self.trace=[]
        self.agent_Env=Env()
        self.agent_End=self.agent_Env.End
        self.prev_state=None
        self.prev_action=None
        self.current_state=None
        self.reward=None
        
    def state_converter(self,q_state):
         """Converts states which are in terms of 3D tuple coordinates in q_tables into single values representation
         """
         single_val=q_state[0]*36+q_state[1]*X_RANGE*36+q_state[2]/10
         return int(single_val)
    
    def q_table_converter(self,q_table):
        """Converts the q table which is a nested dictionary into a np 2D array
        """
        q_array=np.zeros((X_RANGE*Y_RANGE*36,len(self.actions)))
        for idx0,val0 in q_table.items():
            x_state=self.state_converter(idx0)
            for idx1,val1 in val0.items():
                y_action=self.action_dict[idx1]
                value=val1
                q_array[x_state,y_action]=value
        return q_array
    
    
    def feature_matrix_generator(self,final_q_table):
        """This is for input to ANN, converts q table from a nested dictionary to a np matrix with state and policy"""
        feature_list=[]     #nested list to be converted later to the feature matrix
        for idx,val in final_q_table.items():
            policy=self.action_dict[self.argmax(val)]
            feature_list.append([idx[0],idx[1],idx[2],policy])
        return np.array(feature_list)
    
    def take_action(self,action):
        self.current_state=self.agent_Env.env_step(action)[1]
        self.reward=self.agent_Env.env_step(action)[0]
        self.agent_End=self.agent_Env.env_step(action)[2]
        updated_Env=Env(self.current_state)                  #update the environment with the action selected
        return updated_Env
    
    def agent_play(self,num_episodes):
        """Input the number of training episodes, agent will move and update the q table"""
        counter=0
        sum_reward=0
        while (counter<num_episodes):
            ini_act=self.agent_start(START)
            self.agent_Env=self.take_action(ini_act)
            self.trace.append([self.current_state,ini_act])
            sum_reward+=self.reward
            #print('agent picks',ini_act,'to reach',self.current_state)
            #inner_count=0
            while self.reward!=1000:
                act=self.agent_step(reward=self.reward,state=self.current_state)
                self.agent_Env=self.take_action(act)
                self.trace.append([self.current_state,act])
                sum_reward+=self.reward
                #print('agent picks',act,'to reach',self.current_state)
#                 if inner_count>=10000:
#                     break
#                 inner_count+=1
            #print('agent terminates at',self.current_state,'with reward',self.reward,' episode ends?',self.agent_End)
            if self.agent_End==True:
                print('agent terminates at',self.current_state,'with reward',self.reward)
                self.agent_end(reward=1000)
            if counter<num_episodes-1:
                self.reset()    #reset the environment for the next episode, but keep the final trace for visualisation
            self.sum_reward_list.append(sum_reward)
            sum_reward=0
            counter+=1
        pass

def write_csv_file(csv_table, file_name, file_delimiter, quoting_value):
    with open(file_name, 'a', newline='') as csv_file:
        csv_writer=csv.writer(csv_file,delimiter=file_delimiter,quoting=quoting_value)
        for row in csv_table:
            csv_writer.writerow(row)
    
if __name__=="__main__":
    testq=Q_agent()
    testq.agent_play(30000)
    state_trace=[]
    xy_trace=[]
    action_list=[]
    for state,action in testq.trace:
        xy_trace.append(state[:-1])
        state_trace.append(state)
        action_list.append(action)
        print(action)
    state_trace.append(WIN_AVR)
    xy_trace.append((WIN_AVR[:-1]))
    print("agent's current path:\n")
    Env().show_grid(final=state_trace,final_xy=xy_trace,show_ship=True)
    plt.plot(list(range(500,len(testq.sum_reward_list))),testq.sum_reward_list[500:])
    plt.xlabel('episode number')
    plt.ylabel('sum of rewards per episode')
    plt.show()
    #write_csv_file(csv_table=[xy_trace,state_trace,action_list,testq.sum_reward_list],file_name='csv_single_obs_data', file_delimiter=',', quoting_value=csv.QUOTE_NONNUMERIC)