import pandas as pd
import numpy as np
#purchase_1 = pd.Series({'Name': 'Chris',
#                        'Item Purchased': 'Dog Food',
#                        'Cost': 22.50})
#purchase_2 = pd.Series({'Name': 'Kevyn',
#                        'Item Purchased': 'Kitty Litter',
#                        'Cost': 2.50})
#purchase_3 = pd.Series({'Name': 'Vinod',
#                        'Item Purchased': 'Bird Seed',
#                        'Cost': 5.00})
#
#df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
#print(df)
#
#boolean_mask=(df['Cost']>=2.60)
#print(boolean_mask)
#
#masked_df=df.where(boolean_mask)   #map the boolean mask to the dataframe, where the False boolean value will lead to a NaN row
#print(masked_df)
#
#masked_df=masked_df.dropna()       #drop the row with NaN value
#print(masked_df.head())
#
##masked_df1=(df)[boolean_mask]      #use boolean mask to index, same effect as above lines, but simpler
##print(masked_df1)
#
#complex_df=df[(df['Cost']>=3.00) | (df['Name']=='Kevyn')]      #use multiple boolean masks comparing to create a single boolean mask and map to the dataframe
#print(complex_df)                                              #!!! use the '|'and '&' symbol instead of 'or' and 'and' to avoid error when comparing series of boolean




purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
print(df)

df['Location']=df.index
df=df.set_index(['Location','Name'])              #pass a list of indices to set_index creates a row hierarchy
print(df.head())
df=df.append(pd.Series({'Item Purchased': 'Kitty Food','Cost': 3.00},name=('Store 2','Kevyn')))    #append doesn't mutate the original dataframe, thus an update is needed
print(df)                                                           #the name argument must be within a tuple or immutable object

print(df.index)
print('DogFood'.isalpha())