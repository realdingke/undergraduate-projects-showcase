import numpy as np
"""This algorithm is built based on the structure of Jeremy Zhang's concise but intuitive gridworld implemnetation,
   available at https://towardsdatascience.com/implement-grid-world-with-q-learning-51151747b455 ,
   so credit goes to him.
   This is a demonstration of Q learning agent obstacle avoidance in a small gridworld scenario,
   assume deterministic action, agent uses epsilon-greedy action selection,
   with a grid layout as such:
   -----------------
   0 | 1 | 2|  3|...
   1 |
   2 |
   ...
   each coordinate is expressed as (i,j), where i is row index, j is column index,
   After the Q agent class, there is also an Expected Sarsa agent class implemented, performances are compared.
"""
#global variables
GRID_ROWS=5                #different row and column numbers to break the symmetry, allows easy occurence of optimal path to goal
GRID_COLS=6
START=(GRID_ROWS-2,0)      #the starting position
WIN=(0,GRID_COLS-1)        #the winning position that ends an epsiode

def obs():
    """generates random number of obstacles in random (i,j) position,there is a maximum number of obstacles
       (otherwise the agent might not be able to end episode and backpropagate the reward)
       returns a list of tuples of obstacle position
    """
    obs_list=[]
    for num in range(np.random.choice([n for n in range(1,max(GRID_ROWS,GRID_COLS))])):
        i=np.random.choice([n for n in range(GRID_ROWS)])
        j=np.random.choice([n for n in range(GRID_COLS)])
        if (i,j) not in obs_list:
            if (i,j)!=START and (i,j)!=WIN:
                obs_list.append((i,j))
    return obs_list
OBS_LIST=obs()



#To simulate the agent-environment interation, two separate classes are defined
class Env():                         #the environment class
    def __init__(self,state=START):
        self.grid=np.zeros([GRID_ROWS,GRID_COLS])     #setup the grid
        self.obstacles=OBS_LIST
        #self.obstacles=[(0,2),(2,2)]            #can overwrite manually
        if len(self.obstacles)>0:
            for tup in self.obstacles:
                self.grid[tup]=-1
        self.state=state
        self.End=False
    
    
    def env_step(self,action):
        """The environment decides the step,
           argument is an action taken: could be one of "up","down","left","right",
           returns the next position on grid
        """
        if action=='up':
            possible_next_state=(self.state[0]-1,self.state[1])
        elif action=='down':
            possible_next_state=(self.state[0]+1,self.state[1])
        elif action=='left':
            possible_next_state=(self.state[0],self.state[1]-1)
        elif action=='right':
            possible_next_state=(self.state[0],self.state[1]+1)
        else:
            raise Exception(str(action)+'is not in available actions!')
        #check if possible next state is legal
        if possible_next_state[0]>=0 and possible_next_state[0]<GRID_ROWS:
            if possible_next_state[1]>=0 and possible_next_state[1]<GRID_COLS:
                if possible_next_state not in self.obstacles:
                    next_state=possible_next_state
                    return next_state
        return self.state
           
    
    def env_reward(self):
        """Only return reward of 100 if agent reaches target
           all other moves receive 0 reward
        """
        if self.state==WIN:
            return 100
        return 0
    
    def isitEnd(self):
        if self.state==WIN:
            self.End=True

    def showgrid(self,onlygrid=False):
        if not onlygrid:
            self.grid[self.state]=1
            self.grid[WIN]=2
        for i in range(GRID_ROWS):
            print('-'*4*GRID_COLS+'-')
            gridrow='|'
            for j in range(GRID_COLS):
                if self.grid[i,j]==1:
                    indicator=' *'
                if self.grid[i,j]==-1:
                    indicator=' X'
                if self.grid[i,j]==0:
                    indicator=' 0'
                if self.grid[i,j]==2:
                    indicator=' ^'
                gridrow+=indicator+' |'
            print(gridrow)
        print('-'*(4*GRID_COLS+1))
        
Env().showgrid()         
            
#the Q agent class
class Q_agent():
    def __init__(self):
        self.agent_Env=Env()
        self.actions=['up','down','left','right']
        self.agent_End=self.agent_Env.End
        self.step_size=0.2
        self.discount=0.9
        self.epsilon=0.1
        self.q_values={}         #action value q(s,a)
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                self.q_values[(i,j)]={}
                for action in self.actions:
                    self.q_values[(i,j)][action]=0          #initialize q_values as a nested dictionary, state(in coordinate tuple) mapped to its actions, each action mapped to a q value
        self.trace=[]            #a record of position and action pairs
    
    def choose_action(self):
        """the agent's behaviour policy is set to be epsilon greedy
        """
        max_q_set=float('-inf')
        if np.random.uniform(0,1)<self.epsilon:
            action=np.random.choice(self.actions)           #agent takes exploratroy actions
        else:
            for act in self.actions:
                current_position=self.agent_Env.state
                q=self.q_values[current_position][act]
                if q>max_q_set:
                    max_q_set=q
                    action=act                          #agent exploits by choosing the action that maximises q value
        return action
    
    def take_action(self,action):
        next_position=self.agent_Env.env_step(action)
        updated_Env=Env(next_position)                  #update the environment with the action selected
        return updated_Env
    
    def agent_steps(self,num_episodes):
        """Agent is on the move, it selects action and move steps according to the Env class,
           once it has reached termnial state, it back-propagate reward to previous states,
           thus updating their action values,
           since steps other than the one leading to terminal state will receive zero reward,
           the q values will not be updateds until the epsiode has ended.
           The target policy is the maximum of all action values at the state.
        """
        counter=0
        while counter<num_episodes:
            if not self.agent_Env.End:            #agent is trying to reach the goal
                action=self.choose_action()
                self.trace.append([self.agent_Env.state,action])  #record agent's trace of the current episode
                print("current position {0} action {1}".format(self.agent_Env.state, action))
                self.agent_Env=self.take_action(action)     #update both the agent and enviroment after one step
                print('next position:',self.agent_Env.state,'\n')
                self.agent_Env.isitEnd()
                self.agent_End=self.agent_Env.End           #unify the epsiode status
            else:       #agent has reached the goal, current epsisode has ended 
                q=self.agent_Env.env_reward()
                for act in self.actions:
                    self.q_values[self.agent_Env.state][act]=q     #no action value difference when at terminal state
                for state,action in reversed(self.trace):   #to backpropagate the action value
                    q=self.q_values[state][action]+self.step_size*(self.discount*q-self.q_values[state][action])  #use temporal difference
                    self.q_values[state][action]=q     #update the q value
                if counter<num_episodes-1:
                    self.reset()    #reset the environment for the next game, but keep the final trace for visualisation
                counter+=1
        pass
    
    def reset(self):
        self.trace=[]
        self.agent_Env=Env()
        self.agent_End=self.agent_Env.End

        

if __name__=="__main__":
    testq=Q_agent()
    testq.agent_steps(50000)
    #print('current Q values:',testq.q_values,'\n')
    print("agent's current path:\n")
    for state,action in testq.trace:
        Env(state).showgrid()
        print(action)
                        
 
 
 #the Expected Sarsa agent class
class Exp_Sarsa_agent():
    def __init__(self):
        self.agent_Env=Env()
        self.actions=['up','down','left','right']
        self.agent_End=self.agent_Env.End
        self.step_size=0.2
        self.discount=0.9
        self.epsilon=0.1
        self.q_values={}         #action value q(s,a)
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                self.q_values[(i,j)]={}
                for action in self.actions:
                    self.q_values[(i,j)][action]=0          #initialize q_values as a nested dictionary, state(in coordinate tuple) mapped to its actions, each action mapped to a q value
        self.trace=[]            #a record of position and action pairs
    
    def choose_action(self):
        """the agent's behaviour policy is set to be epsilon greedy
        """
        max_q_set=float('-inf')
        if np.random.uniform(0,1)<self.epsilon:
            action=np.random.choice(self.actions)           #agent takes exploratroy actions
        else:
            for act in self.actions:
                current_position=self.agent_Env.state
                q=self.q_values[current_position][act]
                if q>max_q_set:
                    max_q_set=q
                    action=act                          #agent exploits by choosing the action that maximises q value
        return action
    
    def take_action(self,action):
        next_position=self.agent_Env.env_step(action)
        updated_Env=Env(next_position)                  #update the environment with the action selected
        return updated_Env
    
    def agent_steps(self,num_episodes):
        """Agent is on the move, it selects action and move steps according to the Env class,
           once it has reached termnial state, it back-propagate reward to previous states,
           thus updating their action values,
           since steps other than the one leading to terminal state will receive zero reward,
           the q values will not be updateds until the epsiode has ended.
           The target policy is the expected value of action values at the state.
        """
        counter=0
        while counter<num_episodes:
            if not self.agent_Env.End:            #agent is trying to reach the goal
                action=self.choose_action()
                self.trace.append([self.agent_Env.state,action])  #record agent's trace of the current episode
                #print("current position {0} action {1}".format(self.agent_Env.state, action))
                self.agent_Env=self.take_action(action)     #update both the agent and enviroment after one step
                #print('next position:',self.agent_Env.state,'\n')
                self.agent_Env.isitEnd()
                self.agent_End=self.agent_Env.End           #unify the epsiode status
            else:       #agent has reached the goal, current epsisode has ended 
                q=self.agent_Env.env_reward()
                for act in self.actions:
                    self.q_values[self.agent_Env.state][act]=q     #no action value difference when at terminal state
                for state,action in reversed(self.trace):   #to backpropagate the action value
                    current_q=self.q_values[state]  
                    non_q=0
                    for idx,val in current_q.items():
                        if not idx==action:             #non_q will be the expectation for all non-greedy action values
                            non_q+=(self.epsilon/len(self.actions))*val       
                    exp_q=(1-self.epsilon+self.epsilon/len(self.actions))*q+non_q   #combine with expection for greedy action value
                    q=self.q_values[state][action]+self.step_size*(self.discount*exp_q-self.q_values[state][action])
                    self.q_values[state][action]=q    #update the action value
                if counter<num_episodes-1:
                    self.reset()    #reset the environment for the next game, but keep the final trace for visualisation
                counter+=1
        pass
    
    def reset(self):
        self.trace=[]
        self.agent_Env=Env()
        self.agent_End=self.agent_Env.End

        

if __name__=="__main__":
    teste=Exp_Sarsa_agent()
    teste.agent_steps(50000)
    #print('current action values:',teste.q_values,'\n')
    print("agent's current path:\n")
    for state,action in teste.trace:
        Env(state).showgrid()
        print(action)
                