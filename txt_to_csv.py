import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCK_YOU = os.path.join("datasets","rockyou.txt")

with open (ROCK_YOU, "r", encoding="latin-1") as f:
    passwords = f.read().splitlines()

df = pd.DataFrame(passwords, columns=["password"])
df.to_csv("rockyou.csv", index=False)
