def convert_dict2list(gpadict):
    """
    This function takes a dictonary and converts it to a list of tuples
    """
    list0=[]
    for (key,value) in gpadict.items():
        list0.append((key,value))
    return list0

gpadict0={"TamikaBarker":3.9,"ElmerBrasher":2.8,"RaymondHathaway":3.3,"RebekahBailey":3.5}
print(convert_dict2list(gpadict0))
        