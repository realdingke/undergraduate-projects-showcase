"""
Template - Select a specific item in a nested list
"""

# Define a nested list of lists
nested_list = [[col + 3 * row for col in range(3)] for row in range(5)]
print(nested_list)

# Add code to print out the item in this nested list with value 7
# To find col+3*row==7, row can be 2 and col to be 1, thus:
print(nested_list[2][1])

# Output
#[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
#7