"""
Project for Week 4 of "Python Data Representations".
Find differences in file contents.

Be sure to read the project description page for further information
about the expected behavior of the program.
"""

IDENTICAL = -1

def singleline_diff(line1, line2):
    """
    Inputs:
      line1 - first single line string
      line2 - second single line string
    Output:
      Returns the index where the first difference between
      line1 and line2 occurs.

      Returns IDENTICAL if the two lines are the same.
    """
    short_length=min(len(line1),len(line2));
    if short_length>0:
        for index in range(short_length):
            if line1[index]!=line2[index]:
                return index 
            else:
                if len(line1)==len(line2) and index==(short_length-1):
                    return IDENTICAL
                elif len(line1)!=len(line2) and index==(short_length-1):
                    return short_length
    elif short_length==0:
        if len(line1)==len(line2):
            return IDENTICAL
        else:
            return 0

#print(singleline_diff("abcdg","abcd"))

def singleline_diff_format(line1, line2, idx):
    """
    Inputs:
      line1 - first single line string
      line2 - second single line string
      idx   - index at which to indicate difference
    Output:
      Returns a three line formatted string showing the location
      of the first difference between line1 and line2.

      If either input line contains a newline or carriage return,
      then returns an empty string.

      If idx is not a valid index, then returns an empty string.
    """
    boolean1=("\n" in line1)or("\r" in line1);
    boolean2=("\n" in line2)or("\r" in line2);
    if (boolean1 or boolean2):
        return ""
    if idx not in range(min(len(line1),len(line2))+1):
        return ""
    else:
        string=line1+"\n"+"="*idx+"^"+"\n"+line2+"\n";
        return string
    
#print(singleline_diff_format("abcdefghj","abcdefg",singleline_diff("abcdefghj","abcdefg")))

def multiline_diff(lines1, lines2):
    """
    Inputs:
      lines1 - list of single line strings
      lines2 - list of single line strings
    Output:
      Returns a tuple containing the line number (starting from 0) and
      the index in that line where the first difference between lines1
      and lines2 occurs.

      Returns (IDENTICAL, IDENTICAL) if the two lists are the same.
    """
    short_listlength=min(len(lines1),len(lines2));
    if short_listlength>0:
        for index in range(short_listlength):
            line_diff_index=singleline_diff(lines1[index],lines2[index]);
            if line_diff_index!=-1:
                tuple1=(index,line_diff_index);
                return tuple1
            elif line_diff_index==-1 and len(lines1)!=len(lines2) and index==short_listlength-1:
                return (short_listlength,0)
            elif line_diff_index==-1 and len(lines1)==len(lines2) and index==short_listlength-1:
                return (IDENTICAL,IDENTICAL)
    elif short_listlength==0:
        if len(lines1)==len(lines2):
            return (IDENTICAL,IDENTICAL)
        else:
            return (0,0)

#print(multiline_diff(["abrd","abd","abc"],["abcd","abd","abc","efg"]))
    


def get_file_lines(filename):
    """
    Inputs:
      filename - name of file to read
    Output:
      Returns a list of lines from the file named filename.  Each
      line will be a single line string with no newline ('\n') or
      return ('\r') characters.

      If the file does not exist or is not readable, then the
      behavior of this function is undefined.
    """
    openfile=open(filename,"rt")
    linelist=list("");
    for line in openfile.readlines():
        if "\n" in line:
            line=line.replace("\n","");
        if "\r" in line:
            line=line.replace("\r","");
        linelist.append(line);
    openfile.close()
    return linelist
    
#print(get_file_lines("testtext.txt"))
    


def file_diff_format(filename1, filename2):
    """
    Inputs:
      filename1 - name of first file
      filename2 - name of second file
    Output:
      Returns a four line string showing the location of the first
      difference between the two files named by the inputs.

      If the files are identical, the function instead returns the
      string "No differences\n".

      If either file does not exist or is not readable, then the
      behavior of this function is undefined.
    """
    filelist1=get_file_lines(filename1);
    filelist2=get_file_lines(filename2);
    tuple_diff=multiline_diff(filelist1,filelist2);
    if tuple_diff!=(IDENTICAL,IDENTICAL):
        string="Line "+str(tuple_diff[0])+":"+"\n";
        if tuple_diff[0]>=min(len(filelist1),len(filelist2)):
            if len(filelist1)>len(filelist2):
                string=string+singleline_diff_format(filelist1[tuple_diff[0]],"",tuple_diff[1]);
                string=string+"The second file is a prefix of the first file!";
            elif len(filelist1)<len(filelist2):
                string=string+singleline_diff_format("",filelist2[tuple_diff[0]],tuple_diff[1]);
                string=string+"The first file is a prefix of the second file!";
        else:
            string+=singleline_diff_format(filelist1[tuple_diff[0]],filelist2[tuple_diff[0]],tuple_diff[1]);
        return string
    else:
        return "No differences.\n"
    pass

print(file_diff_format("testtext1.txt","testtext2.txt"))
