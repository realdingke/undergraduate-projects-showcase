def count_letters(word_list):
    """ See question description """
    
    ALPHABET = "abcdefghijklmnopqrstuvwxyz"
    totalstring=""
    for word in word_list:
        totalstring+=word
    letter_count = {}
    for letter in ALPHABET:
        letter_count[letter]=totalstring.count(letter)
    count=0
    for letter in ALPHABET:
        if letter_count[letter]>count:
            count=letter_count[letter]
            returnletter=letter
#        else:
#            count=count
    return returnletter

print(count_letters(["heello","worrrld"]))
monty_quote = "listen strange women lying in ponds distributing swords is no basis for a system of government supreme executive power derives from a mandate from the masses not from some farcical aquatic ceremony"

monty_words = monty_quote.split(" ")
print(count_letters(monty_words))
        
            
    
        
        
        
        
        