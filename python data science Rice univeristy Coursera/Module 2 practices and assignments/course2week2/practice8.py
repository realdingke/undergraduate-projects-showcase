# Initial list
list1 = [2, 3, 5, 7, 11, 13]

# Make a copy of list1
list2 = list(list1)

# Print out both lists
print(list1)
print(list2)

# Update the first item in second list to zero
list2[0] = 0

# Print out both lists
print(list1)
print(list2)

# Explain what happens to list1 in a comment
# list1 is unaffected by changes in list2 as it items in list1 do not relate to items in list2 now
# list1 and list2 are two different list, due to list2=list(list1), unlike list2=list1(same list)