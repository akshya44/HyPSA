"""
This file builds the dataset for training.
"""

import os
import random
import string
import numpy as np
import re
import pandas as pd 
from features import extractFeatures
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKYOU_PATH = os.path.join(BASE_DIR, "datasets", "rockyou.txt")

# FIXED FEATURE ORDER
FEATURE_ORDER = [
    "length",
    "maxRepetition",
    "shannonEntropy",
    "sequenceCount",
    "dictionaryScore",
    "keyboardScore",
]

# LOAD ROCKYOU
def load_rockyou(limit=50000):
    passwords = set()
    with open(ROCKYOU_PATH, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            pwd = line.strip()
            if pwd:
                passwords.add(pwd)
            if i >= limit:
                break
    return list(passwords)


# WORD POOL FROM ROCKYOU
def extract_word_pool(passwords, limit=5000):
    words = set()

    for pwd in passwords:
        tokens = re.findall(r'[a-zA-Z]{4,}', pwd.lower())

        for token in tokens:
            words.add(token)

        if len(words) >= limit:
            break

    return list(words)


# TRICKY WEAK PASSWORDS
def generate_tricky_weak_password():
    weak_patterns = ["qwerty", "password", "admin", "iloveyou"]
    symbols = "!@#$%^&*"

    base = random.choice(weak_patterns)

    patterns = [
        base + "@" + str(random.randint(100, 9999)),
        base.capitalize() + "#" + str(random.randint(10, 999)),
        base + random.choice(symbols) + "123",
        base + "2025!",
        base + random.choice(symbols) + random.choice(symbols)
    ]

    return random.choice(patterns)


def generate_tricky_weak_dataset(n=10000):
    return [generate_tricky_weak_password() for _ in range(n)]


# MEDIUM PASSWORDS
def generate_medium_password(word_pool):
    word = random.choice(word_pool)

    patterns = [
        word + str(random.randint(10, 9999)),
        word.capitalize() + "@" + str(random.randint(100, 9999)),
        word + random.choice("!@#$") + str(random.randint(10, 999)),
        word.capitalize() + str(random.randint(1000, 9999)) + "!",
        word + "_" + str(random.randint(100, 999)),
        word + str(random.randint(1, 99)) + word[:2],
    ]

    return random.choice(patterns)


def generate_medium_dataset(word_pool, n=35000):
    return [generate_medium_password(word_pool) for _ in range(n)]


# STRONG PASSWORDS
def generate_strong_password():
    chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}"

    patterns = [
        ''.join(random.choices(chars, k=random.randint(12, 16))),
        ''.join(random.choices(chars, k=10)) + random.choice("!@#$") + str(random.randint(100, 999)),
        ''.join(random.choices(string.ascii_letters, k=5)) +
        ''.join(random.choices(string.digits, k=4)) +
        ''.join(random.choices("!@#$%", k=3)) +
        ''.join(random.choices(string.ascii_letters, k=3)),
        ''.join(random.choices(chars, k=18)),
    ]

    return random.choice(patterns)


def generate_strong_dataset(n=35000):
    return [generate_strong_password() for _ in range(n)]


def mutate_password(pwd):
    substitutions = {
        'a': '@',
        'o': '0',
        'i': '1',
        'e': '3',
        's': '$'
    }

    new_pwd = ""

    for c in pwd:
        if c.lower() in substitutions and random.random() < 0.5:
            new_pwd += substitutions[c.lower()]
        else:
            new_pwd += c

    new_pwd += random.choice("!@#$^&*") + str(random.randint(10,999))

    return new_pwd


# BUILD DATASET
def build_dataset():
    print("Loading weak passwords...")
    weak_passwords = load_rockyou()

    print("Extracting word pool...")
    word_pool = extract_word_pool(weak_passwords)

    print("Generating tricky weak passwords...")
    tricky_weak_passwords = generate_tricky_weak_dataset()

    print("Generating medium passwords...")
    medium_passwords = generate_medium_dataset(word_pool)

    print("Generating strong passwords...")
    strong_passwords = generate_strong_dataset()

    X = []
    y = []

    # Weak (real)
    for pwd in weak_passwords[:15000]:
        features = extractFeatures(pwd)
        X.append([features[f] for f in FEATURE_ORDER])
        y.append(0)

    # Weak (mutated)
    for pwd in weak_passwords[:10000]:
        mutated = mutate_password(pwd)
        features = extractFeatures(mutated)
        X.append([features[f] for f in FEATURE_ORDER])
        y.append(0)

    # Weak (tricky)
    for pwd in tricky_weak_passwords:
        features = extractFeatures(pwd)
        X.append([features[f] for f in FEATURE_ORDER])
        y.append(0)

    # Medium + Strong
    for pwd in medium_passwords:
        features = extractFeatures(pwd)
        X.append([features[f] for f in FEATURE_ORDER])
        y.append(1)

    for pwd in strong_passwords:
        features = extractFeatures(pwd)
        X.append([features[f] for f in FEATURE_ORDER])
        y.append(2)
    
    print("Class Distrbution:", Counter(y))

    # 🔥 FIXED PART (ONLY CHANGE)
    X = pd.DataFrame(X, columns=FEATURE_ORDER)
    y = pd.Series(y, name="label")

    print("Dataset ready!")
    print("Shape:", X.shape)

    return X, y


if __name__ == "__main__":
    build_dataset()