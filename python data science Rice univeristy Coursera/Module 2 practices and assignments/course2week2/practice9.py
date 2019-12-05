def list_max(numbers):
    """
    Given a list of numbers, return the maximum (largest) number
    in the list
    """
    set_value=numbers[0];
    for num in numbers[1:]:
        if num>set_value:
            set_value=num
        pass
    return set_value


# Tests
print(list_max([4]))
print(list_max([-3, 4]))
print(list_max([5, 3, 1, 7, -3, -4]))
print(list_max([1, 2, 3, 4, 5]))