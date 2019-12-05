"""
Solution - Write a function make_dict_lists(length) that returns a dictionary whose keys are in range(length) and whose
corresponding values are lists of zeros with length matching the key
"""


# Add code here
def make_dict_lists(length):
    """
    Given an integer length, return a dictionary whose keys
    lie in range(length) and whose corresponding values are 
    lists of zeros with length matching the key
    """
    my_dict={}
    for num in range(length):
        my_dict[num]=[0 for idx in range(num)]
    return my_dict




# Tests
print(make_dict_lists(0))
print(make_dict_lists(1))
print(make_dict_lists(5))

