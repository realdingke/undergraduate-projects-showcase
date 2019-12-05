NUM_ROWS = 25
NUM_COLS = 25

# construct a matrix
my_matrix = []
for row in range(NUM_ROWS):
    new_row = []
    for col in range(NUM_COLS):
        new_row.append(row * col)
    my_matrix.append(new_row)
#for row in my_matrix:
#    print(row)

def trace1(matrix):
    """
    This function takes a square matrix and returns the sum of items along its main diagonal
    """
    sum_diagonal=0
    for rowindex in range(len(matrix)):
        for colindex in range(len(matrix[rowindex])):
            if rowindex==colindex:
                sum_diagonal+=matrix[rowindex][colindex]
    return sum_diagonal

def trace2(matrix):
    """
    Same as trace1
    """
    sum_list=[row[colindex] for row in matrix for colindex in range(NUM_COLS) if matrix.index(row)==colindex]
    sum_diagonal=0
    for value in sum_list:
        sum_diagonal+=value
    return sum_diagonal

print(trace1(my_matrix))
print(trace2(my_matrix))