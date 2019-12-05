"""
Project for Week 3 of "Python Data Analysis".
Read and write CSV files using a dictionary of dictionaries.

Be sure to read the project description page for further information
about the expected behavior of the program.
"""

import csv

def read_csv_fieldnames(filename, separator, quote):
    """
    Inputs:
      filename  - name of CSV file
      separator - character that separates fields
      quote     - character used to optionally quote fields
    Ouput:
      A list of strings corresponding to the field names in
      the given CSV file.
    """
    with open(filename, 'r', newline='') as csvfile:
        csvfileread=csv.DictReader(csvfile, delimiter=separator, quotechar=quote)
        return csvfileread.fieldnames                   #This method does not read the row of csvfile

#        csvfileread=csv.reader(csvfile, delimiter=separator, quotechar=quote)
#        count=0
#        for row in csvfileread:
#            if count<1:
#                fieldnames=row
#            count+=1
#        return fieldnames                               #The lower method uses csv.reader function(list), and reads the row of csvfile

#print(read_csv_fieldnames("name_table.csv", ",", '"'))


def read_csv_as_list_dict(filename, separator, quote):
    """
    Inputs:
      filename  - name of CSV file
      separator - character that separates fields
      quote     - character used to optionally quote fields
    Output:
      Returns a list of dictionaries where each item in the list
      corresponds to a row in the CSV file.  The dictionaries in the
      list map the field names to the field values for that row.
    """
    with open(filename, 'r', newline='') as csvfile:
        csvfileread=csv.DictReader(csvfile, delimiter=separator, quotechar=quote)
        list_of_dict=[]
        for row in csvfileread:
            list_of_dict.append(row)
    return list_of_dict

#print(read_csv_as_list_dict("test0.csv", ",", '"'))




def read_csv_as_nested_dict(filename, keyfield, separator, quote):
    """
    Inputs:
      filename  - name of CSV file
      keyfield  - field to use as key for rows
      separator - character that separates fields
      quote     - character used to optionally quote fields
    Output:
      Returns a dictionary of dictionaries where the outer dictionary
      maps the value in the key_field to the corresponding row in the
      CSV file.  The inner dictionaries map the field names to the
      field values for that row.
    """
    with open(filename, 'r',  newline='') as csvfile:
        csvfileread=csv.DictReader(csvfile, delimiter=separator, quotechar=quote)
        nested_dict={}
        for row in csvfileread:
            nested_dict[row[keyfield]]=row
        return nested_dict
        

def write_csv_from_list_dict(filename, table, fieldnames, separator, quote):
    """
    Inputs:
      filename   - name of CSV file
      table      - list of dictionaries containing the table to write
      fieldnames - list of strings corresponding to the field names in order
      separator  - character that separates fields
      quote      - character used to optionally quote fields
    Output:
      Writes the table to a CSV file with the name filename, using the
      given fieldnames.  The CSV file should use the given separator and
      quote characters.  All non-numeric fields will be quoted.
    """
    with open(filename,'w',newline='') as csvfile:
        csvfilewrite1=csv.writer(csvfile, delimiter=separator, quotechar=quote, quoting=csv.QUOTE_NONNUMERIC)
        csvfilewrite1.writerow(fieldnames)
        csvfilewrite2=csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator,
                                    quotechar=quote, quoting=csv.QUOTE_NONNUMERIC)
        for row in table:
            csvfilewrite2.writerow(row)
        
    pass

