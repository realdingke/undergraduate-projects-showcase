def strange_sum(numbers):
    """
    This function takes a list of integers and returns the sum of those items in the list that are not divisible by 3.
    """
    total=0;
    for num in numbers:
        if num%3!=0:
            total+=num;
        else:
            pass
    return total

print(strange_sum([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]))
print(strange_sum(list(range(123)) + list(range(77))))