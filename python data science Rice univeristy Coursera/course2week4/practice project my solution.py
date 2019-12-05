"""
Week 4 practice project template for Python Data Representation
Update syntax for print in CodeSkulptor Docs
from "print ..." syntax in Python 2 to "print(...)" syntax for Python 3
"""

# HTML tags that bounds example code
PREFIX = "<pre class='cm'>"
POSTFIX = "</pre>"
PRINT = "print"


def update_line(line):
    """
    Takes a string line representing a single line of code
    and returns a string with print updated
    """
    string1=line.lstrip();
    if "print" in string1:
        string2="print"+"("+string1[6:]+")";
        string3=line[0:line.index("p")]+string2;
        return string3
    else:
        return line

    # Strip left white space using built-in string method lstrip()

    # If line is print statement,  use the format() method to add insert parentheses

    # Note that solution does not handle white space/comments after print statememt

# Some simple tests
#print(update_line(""))
#print(update_line("foobar()"))  
#print(update_line("print 1 + 1"))      
#print(update_line("    print 2, 3, 4"))

#Expect output
#
#foobar()
#print(1 + 1)
#    print(2, 3, 4)


def update_pre_block(pre_block):
    """
    Take a string that correspond to a <pre> block in html and parses it into lines.  
    Returns string corresponding to updated <pre> block with each line
    updated via process_line()
    """
    updated_block="";
    count=0;
    for line in pre_block.splitlines():
        count+=1;
        if count<len(pre_block.splitlines()):
            line=update_line(line)+"\n";
            updated_block=updated_block+line;
        else:
            line=update_line(line);
            updated_block=updated_block+line;
            
    return updated_block

# Some simple tests
#print(update_pre_block(""))
#print(update_pre_block("foobar()"))
#print(update_pre_block("if foo():\n    bar()"))
#print(update_pre_block("print\nprint 1+1\nprint 2, 3, 4"))
#print(update_pre_block("    print a + b\n    print 23 * 34\n        print 1234"))

# Expected output
##
##foobar()
##if foo():
##    bar()
##print()
##print(1+1)
##print(2, 3, 4)
##    print(a + b)
##    print(23 * 34)
##        print(1234)

def update_file(input_file_name, output_file_name):
    """
    Open and read the file specified by the string input_file_name
    Proces the <pre> blocks in the loaded text to update print syntax)
    Write the update text to the file specified by the string output_file_name
    """
    openfile=open(input_file_name,"rt");
    filetext=openfile.read();
    openfile.close()
    text_updated="";
    for line in filetext.split("</pre>"):
        if "pre" in line:
            line_pre=line.replace(line[0:line.rindex(">")+1],"");
            line_updated=update_pre_block(line_pre);
            line_total=line.replace(line[line.rindex(">")+1:],line_updated)+"</pre>";
            text_updated=text_updated+line_total;
        else:
            text_updated=text_updated+line;
    openfile_updated=open(output_file_name,"wt");
    openfile_updated.write(text_updated)
    openfile_updated.close()
    pass
    
    # open file and read text in file as a string

    # split text in <pre> blocks and update using update_pre_block()

    # Write the answer in the specified output file
    

# A couple of test files
update_file("table.htm", "table_updated.htm")
update_file("docs.htm", "docs_updated.htm")

# Import some code to check whether the computed files are correct
import examples3_file_diff as file_diff
file_diff.compare_files("table_updated.htm", "table_updated_solution.htm")
file_diff.compare_files("docs_updated.htm", "docs_updated_solution.htm")

# Expected output
##table_updated.html and table_updated_solution.html are the same
##docs_updated.html and docs_updated_solution.html are the same