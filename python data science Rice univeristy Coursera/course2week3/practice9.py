def flatten(nested_list):
    """
    Given a list whose items are list, 
    return the list formed by joining all of these lists
    """
    combined_list=[];
    for item in nested_list:
        combined_list=combined_list+list(item); # or use combined_list.extend(item) to break up the sublist and add to the combined list
    return combined_list

print(flatten([]))
print(flatten([[]]))
print(flatten([[1, 2, 3]]))
print(flatten([["cat", "dog"], ["pig", "cow"]]))
print(flatten([[9, 8, 7], [6, 5], [4, 3, 2], [1]]))

