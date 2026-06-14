import math

# def theoreticalEntropy (password):
#     #Defining the character set:
#     lowercase = set ("abcdefghijklmnopqrstuvwxyz")
#     uppercase = set ("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#     digits = set ("0123456789")
#     symbols = set ("!@#$%^&*()-_=+[]{}|;:'\"<>,.?/")
    
#     #Calculating the size of the character set:
#     charset_size = 0
#     if any (char in lowercase for char in password):
#         charset_size += 26
#     if any (char in uppercase for char in password):
#         charset_size += 26
#     if any (char in digits for char in password):
#         charset_size += 10
#     if any (char in symbols for char in password):
#         charset_size += len(symbols)
    
#     #Calculating the entropy:
#     if charset_size == 0:
#         return 0
#     entropy = len(password) * math.log2(charset_size)
#     return entropy

def shannonEntropy(password):
    if len(password) == 0:
        return 0

    char_freq = {}
    for char in password:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1

    entropy = 0
    for freq in char_freq.values():
        p = freq / len(password)
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


# if __name__ == "__main__":
#     password = input("Enter a password to calculate its theoretical entropy: ")
#     theoreticalEntropy_score = theoteticalEntropy(password)
#     shannonEntropy_score = shannonEntropy(password)
#     print(f"The theoretical entropy of the password '{password}' is: {theoreticalEntropy_score:.2f} bits")
#     print(f"The Shannon entropy of the password '{password}' is: {shannonEntropy_score:.2f} bits")