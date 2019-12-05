#myList = [1, 2, 3, 1, 2, 5, 6, 7, 8]
#cleanlist = []
#[cleanlist.append(x) for x in myList if x not in cleanlist]


def list_rindex(list_in,item_name):
    """
    This function takes a list and an item and returns the index of the first occurence of the item counting from the right.
    """
    for num in reversed(range(len(list_in))):
        if list_in[num]==item_name:
            return num
    pass
    

def remove_duplicates(items):
    """
    Given a list, return a list with duplicate items removed
    and the remaining items in the same order
    """
    for item in items:
        if items.count(item)>1:
            items.pop(list_rindex(items,item))
        else:
            pass
    return items


print(remove_duplicates([]))
print(remove_duplicates([1, 2, 3, 4]))
print(remove_duplicates([1, 2, 2, 3, 3, 3, 4, 5, 6, 6]))
print(remove_duplicates(["cat", "dog", "cat", "pig", "cow", "cat", "pig", "pug"]))
