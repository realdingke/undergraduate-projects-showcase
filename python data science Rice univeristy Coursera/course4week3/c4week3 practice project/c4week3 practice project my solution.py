"""
Week 3 practice project template for Python Data Visualization
Read two CSV files and join the resulting tables based on shared FIPS codes
Analyze both data sources for anamolous FIPS codes
"""

import csv
import copy



#########################################################
# Provided code for week 3

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
    Output: Nested list consisting of the fields in the CSV file
    """
       
    with open(file_name, newline='') as csv_file:       # don't need to explicitly close the file now
        csv_table = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            csv_table.append(row)
    return csv_table



def write_csv_file(csv_table, file_name):
    """
    Input: Nested list csv_table and a string file_name
    Action: Write fields in csv_table into a comma-separated CSV file with the name file_name
    """
    
    with open(file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for row in csv_table:
            csv_writer.writerow(row)



# Part 1 - function that creates a dictionary from a table

def make_dict(table, key_col):
    """
    Given a 2D table (list of lists) and a column index key_col,
    return a dictionary whose keys are entries of specified column
    and whose values are lists consisting of the remaining row entries
    """
    return_dict={}
    table0=copy.deepcopy(table)     #To make a copy of a nested list, copy.deepcopy() must be used to copy the inner list as well, list() does not work!!!
    for row in table0:
        key=row[key_col]
        row.pop(key_col)
        return_dict[key]=row
    return return_dict


def test_make_dict():
    """
    Some tests for make_dict()
    """
    table1 = [[1, 2], [3, 4], [5, 6]]
    print(make_dict(table1, 0))
    print(make_dict(table1, 1))
    table2 = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
    print(make_dict(table2, 1))
    print(make_dict(table2, 2))
    
#test_make_dict()



# Part 2 - script for merging the CSV files

CANCER_RISK_FIPS_COL = 2
CENTER_FIPS_COL = 0

def merge_csv_files(cancer_csv_file, center_csv_file, joined_csv_file):
    """
    Read two specified CSV files as tables
    Join the these tables by shared FIPS codes
    Write the resulting joined table as the specified file
    Analyze for problematic FIPS codes
    """
    
    # Read in both CSV files
    nested_list0=read_csv_file(cancer_csv_file)
    nested_list1=read_csv_file(center_csv_file)
    dict0=make_dict(nested_list0,CANCER_RISK_FIPS_COL)
    dict1=make_dict(nested_list1,CENTER_FIPS_COL)
    # Compute joined table, print warning about cancer-risk FIPS codes that are not in USA map
    nested_list2=[#["State","County name","FIPS code","Population","Cancer risk",
                  #"Horizontal coordinate of county center (with respect to the USA SVG map)",
                  #"Vertical coordinate of county center (with respect to the USA SVG map)"]
                 ]
    for key in dict0:
        list3=[]
        if key in dict1:
            list3.extend((dict0[key])[:CANCER_RISK_FIPS_COL])
            list3.append(key)
            list3.extend((dict0[key])[CANCER_RISK_FIPS_COL:])
            list3.extend(dict1[key])
        else:
            print("FIPS code "+str(key)+" is not in USA map file")
            continue                                              #continue returns the code to the loop position and continues the next iteration
        nested_list2.append(list3)
    # Write joined table
    write_csv_file(nested_list2,joined_csv_file)
    # Print warning about FIPS codes in USA map that are missing from cancer risk data
    for key in dict1:
        if key not in dict0:
            print("FIPS code "+str(key)+" is not in cancer risk data")
    pass



merge_csv_files("cancer_risk_trimmed.csv", "USA_Counties_with_FIPS_and_centers.csv", "cancer_risk_joined.csv")




## Part 3 - Explanation for anomalous FIPS codes
## https://www1.udel.edu/johnmack/frec682/fips_codes.html
##
## Output anamolies for cancer risk data
## Puerto Rico, Virgin Island, Statewide, Nationwide - FIPS codes are all not present on USA map
## One specific county (Clifton Forge, VA - 51560) is also not present in USA map.
## According URL above, Clifton Forge was merged with another VA county prior to 2001
##
## Output anamolies for USA map
## State_Line, separator - FIPS codes are all not present in cancer-risk data
## One specific county (Broomfield County - 08014) is also not present in cancer-risk data
## Accoring to URL above, Broomfield County was created in 2001
##
## Implies cancer risk FIPS codes were defined prior to 2001, the USA map FIPS codes were defined after 2001
