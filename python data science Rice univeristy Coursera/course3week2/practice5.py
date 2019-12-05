"""
Solution - Analyze a reference issue involving a nested list
"""

# Create a nested list
zero_list = [0, 2, 0]
nested_list = []
for dummy_idx in range(5):
    #nested_list.append(zero_list)
    nested_list.append(list(zero_list)) 
print(nested_list)
    
# Update an entry to be non-zero
nested_list[2][1] = 7
print(nested_list)


# Erroneous output
#[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
#[[0, 7, 0], [0, 7, 0], [0, 7, 0], [0, 7, 0], [0, 7, 0]]

# Desired output
# [[0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 2, 0]]
# [[0, 2, 0], [0, 2, 0], [0, 7, 0], [0, 2, 0], [0, 2, 0]]

#To solve this, use list(zero_list) to create a copy of the zero_list and this cancels the reference to zero_list.


# Explanation:
# The nested list was created by refering to the same zero list, the nested_list[2][1]=7 updates the zero list therefore change all sublists of nested list.