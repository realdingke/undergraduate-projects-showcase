"""
Week 3 practice project template for Python Data Analysis
Reading and writing CSV files using lists
"""


import csv



#########################################################
# Part 1 - Week 3



def print_table(table):
    """
    Echo a nested list to the console
    """
    for row in table:
        print(row)


def read_csv_file(file_name):
    """
    Given a CSV file, read the data into a nested list
    Input: String corresponding to comma-separated  CSV file
    Output: Lists of lists consisting of the fields in the CSV file
    """
    with open(file_name, "r", newline='') as csvfile:     #"with as" allows not to use close() everytime
        csvfileread=csv.reader(csvfile, delimiter=",")
        csvtable=[]
        for row in csvfileread:            #Each row read from the csv file is returned as a list of strings!
            csvtable.append(row)
    return csvtable



def write_csv_file(csv_table, file_name):
    """
    Input: Nested list csv_table and a string file_name
    Action: Write fields in csv_table into a comma-separated CSV file with the name file_name
    """
    with open(file_name,"w",newline='') as csvfile:
        csvfilewrite=csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for listrow in csv_table:
            csvfilewrite.writerow(listrow)
    pass

        
def test_part1_code():
    """
    Run examples that test the functions for part 1
    """
    
    # Simple test for reader
    test_table = read_csv_file("test_case.csv")  # create a small CSV for this test
    print_table(test_table)
    print(test_table[0][0])
    print()

    # Test the writer
    cancer_risk_table = read_csv_file("cancer_risk05_v4_county.csv")
    write_csv_file(cancer_risk_table, "cancer_risk05_v4_county_copy.csv")
    cancer_risk_copy = read_csv_file("cancer_risk05_v4_county_copy.csv")
    
    # Test whether two tables are the same
    for row in range(len(cancer_risk_table)):
        for col in range(len(cancer_risk_table[0])):
            if cancer_risk_table[row][col] != cancer_risk_copy[row][col]:
                return "Difference at"+str(row)+str(col)+str(cancer_risk_table[row][col])+str(cancer_risk_copy[row][col])
    
    return "Correct"


print(test_part1_code())
table0=read_csv_file("test_case.csv")
write_csv_file(table0, "test1.csv")

