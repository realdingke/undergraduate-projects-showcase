def aggregate_by_player_id(statistics, playerid, fields):
    """
    Inputs:
      statistics - List of batting statistics dictionaries
      playerid   - Player ID field name
      fields     - List of fields to aggregate
    Output:
      Returns a nested dictionary whose keys are player IDs and whose values
      are dictionaries of aggregated stats.  Only the fields from the fields
      input will be aggregated in the aggregated stats dictionaries.
    """
    keylist=[]                                      #The list of playerids
    for stats in statistics:
        if stats[playerid] not in keylist:
            keylist.append(stats[playerid])
    nested_dict={}
    for name in keylist:
        inner_dict={}
        for fieldname in fields:
            inner_dict[fieldname]=0
        for stats in statistics:
            for fieldname in fields:
                if name==stats[playerid]:
                    inner_dict[fieldname]+=int(stats[fieldname])
                    inner_dict[playerid]=stats[playerid]
        if name==inner_dict[playerid]:
            nested_dict[name]=inner_dict
    return nested_dict

print(aggregate_by_player_id([{'player': '1', 'stat1': '3', 'stat2': '4', 'stat3': '5'},
{'player': '1', 'stat1': '2', 'stat2': '1', 'stat3': '8'},
{'player': '1', 'stat1': '5', 'stat2': '7', 'stat3': '4'}],
'player', ['stat1']))