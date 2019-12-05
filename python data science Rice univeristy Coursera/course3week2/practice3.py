matrix =[
[1, 2, 3, 4],
[5, 6, 7, 8],
[9, 10, 11, 12],
]
def matrix_transpose1(matrix):
    transposed_matrix_list=[]
    rowlength=len(matrix[0])
    for rowindex in range(rowlength):
        column_list=[]
        for column in matrix:
            column_list.append(column[rowindex])
        transposed_matrix_list.append(column_list)
    return transposed_matrix_list

print(matrix_transpose1(matrix))
        
def matrix_transpose2(matrix):
    return [[column[rowindex] for column in matrix] for rowindex in range(len(matrix[0]))]

print(matrix_transpose2(matrix))



# Add code here for a list comprehension
zero_list=[0 for i in range(3)]
# Add code here for nested list comprehension
nested_list=[zero_list for i in range(5)]

# Tests
print(zero_list)
print(nested_list)

# Output
#[0, 0, 0]
#[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        
        