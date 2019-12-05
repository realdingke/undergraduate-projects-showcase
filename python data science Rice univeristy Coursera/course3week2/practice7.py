"""
Template - Write a function dict_copies(my_dict, num_copies) that 
returns a list consisting of num_copies copies of my_dict
"""


# Add code here
def dict_copies(my_dict, num_copies):
    """
    Given a dictionary my_dict and an integer num_copies, 
    returns a list consisting of num_copies copies of my_dict.
    """
    return [dict(my_dict) for num in range(num_copies)]           # add dict(my_dict) to make copies of my_dict to avoid referencing problems

# Tests
print(dict_copies({}, 0))
print(dict_copies({}, 1))
print(dict_copies({}, 2))

test_dict = dict_copies({'a' : 1, 'b' : 2}, 2)
print(test_dict)

# Check for reference problem
test_dict[1]["a"] = 3
print(test_dict)



# Output
#[]
#[{}]
#[{}, {}]
#[{'a': 1, 'b': 2}, {'b': 2, 'a': 1}]
#[{'b': 2, 'a': 1}, {'b': 2, 'a': 3}]