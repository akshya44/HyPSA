#from entropy import theoreticalEntropy, shannonEntropy 
"""
|
-> Commented out since we are importing these functions in features.py and using them in the 
   extractFeatures function. No need to import them again here.
"""
from features import extractFeatures
def analyzePassword(password):
    features = extractFeatures(password)
    print(f"Analyzing Password: {password}")
    for key, value in features.items():
        print(f"{key}: {value}")
    
if __name__ == "__main__":
    # analyzePassword("SaSw@T1234321")
    # analyzePassword("P@$$w0rd123!")
    analyzePassword("p@$$w0rd")