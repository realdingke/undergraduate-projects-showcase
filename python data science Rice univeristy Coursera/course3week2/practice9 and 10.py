"""
Template - Create a dictionary grade_table whose keys are provided
student names and values are associated list of grades
"""


# Add code here
grade_table = {
    "Joe":[100,98,100,13],
    "Scott":[75,59,89,77],
    "John":[86,84,91,78]
}



# Add code here

def make_grade_table(name_list, grades_list):
    """
    Given a list of name_list (as strings) and a list of grades
    for each name, return a dictionary whose keys are
    the names and whose associated values are the lists of grades
    """
    my_dict={}
    zippednamegrades=zip(name_list,grades_list)  #zip creates a tuple that pairs the name and corresponding grades
    for item in zippednamegrades:
        my_dict[item[0]]=item[1]
    return my_dict
        

# Tests
print(make_grade_table([], []))

name_list = ["Joe", "Scott", "John"]
grades_list = [100, 98, 100, 13], [75, 59, 89, 77],[86, 84, 91, 78] 
print(make_grade_table(name_list, grades_list))
