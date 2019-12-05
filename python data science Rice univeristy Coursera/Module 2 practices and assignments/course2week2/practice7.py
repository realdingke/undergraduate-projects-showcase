# Initial list
list1 = [2, 3, 5, 7, 11, 13]

# Another reference to list1
list2 = list1

# Print out both lists
#print(list1)
#print(list2)

# Update the first item in second list to zero
list2[0] = 0

# Print out both lists
print(list1)
print(list2)

# Explain what happens to list1 in a comment
# List1 is automatically updated when list2 is changed
