NUM_ROWS = 5
NUM_COLS = 9

# construct a matrix
my_matrix = {}
for row in range(NUM_ROWS):
    row_dict = {}
    for col in range(NUM_COLS):
        row_dict[col] = row * col
    my_matrix[row] = row_dict
    
print(my_matrix)
 
# print the matrix
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        print(my_matrix[row][col], end = " ")
    print()

print(my_matrix=={1: {7: 7, 4: 4, 3: 3, 8: 8, 6: 6, 5: 5, 2: 2, 0: 0, 1: 1}, 0: {0: 0, 7: 0, 3: 0, 4: 0, 8: 0, 6: 0, 5: 0, 1: 0, 2: 0}, 2: {0: 0, 8: 16, 5: 10, 2: 4, 7: 14, 4: 8, 1: 2, 3: 6, 6: 12}, 3: {1: 3, 7: 21, 2: 6, 8: 24, 3: 9, 4: 12, 6: 18, 0: 0, 5: 18}, 4: {3: 12, 7: 28, 0: 0, 2: 8, 1: 4, 4: 16, 6: 24, 5: 20, 8: 32}})