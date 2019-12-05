def concatenate_ints(int_list):
    """
    Given a list of integers int_list, return the integer formed by
    concatenating their decimal digits together
    """
    strings_accumulation=str(int_list[0]);
    for num in int_list[1:len(int_list)]:
        strings_accumulation=strings_accumulation+str(num);
    concatenation=int(strings_accumulation);
    return concatenation

# Tests
#print(concatenate_ints([4]))
#print(concatenate_ints([4, 0, 4]))
#print(concatenate_ints([123, 456, 789]))
#print(concatenate_ints([32, 796, 1000]))
print(concatenate_ints([123,456,789,1011,1213,14151617181920,999]))
#list1=[10,21,33];
#print(int(str(list1[0])+str(list1[1])+str(list1[2])))