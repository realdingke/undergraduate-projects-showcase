import random

def make_cipher_dict(alphabet):
    """
    Given a string of unique characters, compute a random 
    cipher dictionary for these characters
    """
#    alphabetlist=list()
#    for num in range(ord("a"),(ord("z")+1)):
#        alphabetlist.append(chr(num))
#    random.shuffle(alphabetlist)
#    tuplelist=list()
#    for char in alphabet:
#        singletuple=(char,alphabetlist[random.randint(0,25)])       #potential repeated value assignments!!!
#        tuplelist.append(singletuple)
#    return dict(tuplelist)

    alphabetlist=list(alphabet)
    randomlist=list(alphabet)
    random.shuffle(randomlist)
    tuplelist=list()
    for char in alphabet:
        singletuple=(char,randomlist[alphabetlist.index(char)])
        tuplelist.append(singletuple)
    return dict(tuplelist)
# solution:
#    alphabetlist=list(alphabet)
#    randomlist=list(alphabet)
#    random.shuffle(randomlist)
#    dict1=dict()
#    for num in range(len(alphabetlist)):
#        dict1[alphabetlist[num]]=randomlist[num]              #use index to iterate to ensure no repeated value assignments!!!
#    return dict1
# Tests
print("Output for part 3")
print(make_cipher_dict(""))
print(make_cipher_dict("cat"))
print(make_cipher_dict("abcdefghijklmnopqrstuvwxyz "))

# Output for part 3 -  note that answers are randomized
#{}
#{'a': 'a', 't': 'c', 'c': 't'}
#{'a': 'h', 'l': 'u', 'u': 'q', 'b': 'v', 'y': 'a', 'm': 'r', 'p': 'j', 'k': 'e', 'n': 'p', 't': 'x', 'd': 'o', 'c': 'c', 'w': ' ', 'f': 'd', 'r': 'z', 'v': 'l', 's': 'y', 'e': 'b', 'o': 'i', 'x': 'm', 'h': 's', 'i': 'w', 'q': 'g', 'g': 'n', 'j': 'f', 'z': 'k', ' ': 't'}
