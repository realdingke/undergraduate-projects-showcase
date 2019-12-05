def word_count(text, word):
    """
    Given a string text consist of words separate by spaces and a string word
    (with no spaces), return the number of times that word appears in the text
    """
    list1=text.split(" ");
    number=list1.count(word);
    return number

# Tests

print(word_count("this pigdog is a fine pigdog", "pigdog"))
print(word_count("this pigdog is not a dog", "dog"))
print(word_count("this pigdog is not a pig", "pigdog"))
