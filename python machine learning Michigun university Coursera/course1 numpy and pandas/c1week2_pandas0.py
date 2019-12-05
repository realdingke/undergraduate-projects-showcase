import pandas as pd
list0=[1,2,3,4]
list1=['a','b','c','d']
s0=pd.Series(list1,index=list0)
print(s0)
for key,value in s0.iteritems():    #iterate directly over a series gets the value of dictionary(not key)
    print(key,value)                #iterate over .iteritems() gets the key value pair of dictionary 
dict0={1:1,2:2,3:3}
s1=pd.Series(dict0,index=[1,2,3])  #the index argument overwrites the dict0 keys, and assign NaN value to index that has no entry in dict0
print(s1)
print(s1.sum())
print(s1.iloc[0]+s1.iloc[1])
#purchase_1 = pd.Series({'Name': 'Chris',
#                        'Item Purchased': 'Dog Food',
#                        'Cost': 22.50})
#purchase_2 = pd.Series({'Name': 'Kevyn',
#                        'Item Purchased': 'Kitty Litter',
#                        'Cost': 2.50})
#purchase_3 = pd.Series({'Name': 'Vinod',
#                        'Item Purchased': 'Bird Seed',
#                        'Cost': 5.00})
#df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
#print(df.head())
#
#items=df.loc[:,'Item Purchased']  #use df.loc to index the data first by row then by column
#print(items)
#
##items=df.loc[['Store 1','Store 2'],'Item Purchased']  #same effect as the previous method
##print(items)
#
##items=df.loc[:]['Item Purchased']                     #same effect as the previous method 
##print(items)
#
#print(df.T)   #tansposed dataframe, row becomes column, column becomes row
#
#print(df['Cost'])   #directly access column without using .loc, but cannot direcly access the row
#
#del df['Cost']      #the del operator deletes the column of a dataframe by indexing directy via column head
#print(df)
#
#df1=df.drop('Item Purchased',axis=1)    #the drop() method takes arguments, axis=0 is set to drop row, axis=1 is set to drop column, and inplace=True result in mutating the dataframe, otherwise drop() creates a new dataframe
#print(df1)