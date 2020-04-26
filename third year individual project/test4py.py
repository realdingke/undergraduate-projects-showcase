import numpy as np
def state_converter(q_state):
    """Converts states which are in terms of 3D tuple coordinates in q_tables into single values representation
    """
    single_val=q_state[0]*36+q_state[1]*100*36+q_state[2]/10
    return int(single_val)

def state_back_converter(val):
    """Converts single value representation back to 3D tuple coordinates
    """
    angle=(val%36)*10
    x_coordinate=((val-val%36)/36)%100
    y_coordinate=(val-angle/10-x_coordinate*36)/(100*36)
    return (x_coordinate,y_coordinate,angle)

q_values=[]
for i in range(100):
    for j in range(100):
        for k in range(0,360,10):
            q_values.append((i,j,k))
            
q_single_vals=[state_converter(i) for i in q_values]
q_back_converted_q=[state_back_converter(j) for j in q_single_vals]
print(q_back_converted_q==q_values)

actions=['0','15','35','-15','-35']
action_dict={'-35':0,'-15':1,'0':2,'15':3,'35':4}
q_table={}
for i in range(100):
    for j in range(100):
        for k in range(0,360,10):
            q_table[(i,j,k)]={}
            for action in actions:
                q_table[(i,j,k)][action]=np.random.randint(10)

def feature_matrix_generator(final_q_table):
    """This is for input to ANN, converts q table from a nested dictionary to a np matrix with state and policy"""
    feature_list=[]     #nested list to be converted later to the feature matrix
    for idx,val in final_q_table.items():
        #policy=self.argmax(val)
        feature_list.append([idx[0],idx[1],idx[2]])#,policy])
    return np.array(feature_list)
feature_matrix_generator(q_table)