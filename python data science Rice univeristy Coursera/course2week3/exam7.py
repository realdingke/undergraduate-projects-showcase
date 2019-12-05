"""
Implement the Sieve of Eratosthenes
https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
"""

def compute_primes(bound):
    """
    Return a list of the prime numbers in range(2, bound)
    """
    
    answer = list(range(2, bound))
    for divisor in range(2, bound):
        for num in range(2,bound):
            if num%divisor==0 and num>divisor:
                if num in answer:
                    answer.remove(num)
                else:
                    pass
            else:
                pass 
    return answer

print(len(compute_primes(200)))
print(len(compute_primes(2000)))