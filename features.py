import math
import string
from collections import Counter
from entropy import shannonEntropy
from rapidfuzz import fuzz
import pandas as pd
import re
import os

"""
Feature Engineering Rationale (Updated: 25-03-26)
==================================================
The following features were REMOVED from the final feature set:
  - Lowercase count
  - Uppercase count
  - Digit count
  - Special character count
  - Unique ratio

Reasons:
  1. They are superficial indicators — character composition alone does not
     capture unpredictability or resistance to attacks.
  2. They are already captured by Shannon Entropy, which models character
     diversity and randomness more effectively. Including them causes
     redundancy (double-counting the same information).
  3. They introduce misleading bias — the model learns wrong patterns like
     "more symbols = stronger", rewarding predictable patterns in weak passwords.
  4. Attackers exploit patterns and common substitutions, not raw character counts.
     Layer 3 (dictionary similarity) handles leet-speak substitutions better.
  5. Feature dilution — too many weak features increase noise, reduce model
     clarity, and make interpretation harder.

Final feature set (6 features, all normalized to [0, 1]):
  - length             : Normalized password length (max 16 chars)
  - maxRepetition      : Max character repetition rate
  - shannonEntropy     : Normalized Shannon entropy (actual randomness)
  - sequenceCount      : Normalized sequential pattern score (123, abc, 1357)
  - dictionaryScore    : Fuzzy similarity to common leaked passwords
  - keyboardScore      : Keyboard adjacency chain score (qwerty, asdfgh)

Note on dictionaryScore vs hasDictionaryPattern:
  - dictionaryScore (float, used as ML feature): Uses fuzzy matching via
    RapidFuzz partial_ratio to capture near-matches and substring overlaps.
    It produces a continuous score [0, 1] that the ML models learn from.
  - hasDictionaryPattern (binary, used for rule-based penalties): A fast
    exact substring check against the full dictionary. Returns 1 only when
    a known word of length >= 4 is directly embedded in the password.
    This conservatism is intentional — penalties are hard overrides and should
    only trigger on confirmed matches, not fuzzy near-misses.
  Both functions serve different roles in the hybrid pipeline and are
  deliberately kept separate to avoid over-penalizing edge cases.
"""


# =============================================================================
# LAYER 1: STATISTICAL RANDOMNESS FEATURES
# =============================================================================

def getLength(password):
    """
    Normalized password length.
    Returns a float in [0, 1] where 1.0 means >= 16 characters.
    """
    return min(1.0, len(password) / 16)


def maxRepetitionCount(password):
    """
    Fraction of the password occupied by the most repeated character.
    High values indicate low character diversity.
    Returns 0 for empty passwords.
    """
    length = len(password)
    if length == 0:
        return 0
    char_counts = Counter(password)
    return max(char_counts.values()) / length


def normalize_entropy(password):
    """
    Normalized Shannon entropy: actual entropy divided by the maximum possible
    entropy for the character set used in the password.

    Why Shannon over theoretical entropy?
    Theoretical entropy assumes a uniform distribution across all possible
    characters (overestimating strength). Shannon entropy reflects the actual
    observed frequency distribution in the string.
    """
    if len(password) == 0:
        return 0

    charset = 0
    if any(c.islower() for c in password):
        charset += 26
    if any(c.isupper() for c in password):
        charset += 26
    if any(c.isdigit() for c in password):
        charset += 10
    if any(c in string.punctuation for c in password):
        charset += 32

    max_entropy = math.log2(charset) if charset > 0 else 1
    return min(1.0, shannonEntropy(password) / max_entropy)


# =============================================================================
# LAYER 2: STRUCTURAL / SEQUENTIAL PATTERN FEATURES
# =============================================================================

def maxPatternSequenceCount(password):
    """
    Detects sequential character runs in both ascending and descending order,
    including skip-step patterns (e.g., 1357, aceg).

    Covers:
      - Normal sequences: abc, 123, zyx, 987
      - Skip sequences:   aceg, 1357 (step of 2)

    Returns the normalized length of the longest detected sequence.
    Returns 0 if the longest run is shorter than 3 characters.
    """
    length = len(password)
    if length == 0:
        return 0

    countIncr = 1
    countDecr = 1
    countSkip = 1
    maxCount = 0

    pwd = password.lower()

    for i in range(1, len(pwd)):
        curr = pwd[i]
        prev = pwd[i - 1]

        if (curr.isdigit() and prev.isdigit()) or (curr.isalpha() and prev.isalpha()):
            diff = ord(curr) - ord(prev)

            if diff == 1:
                countIncr += 1
                countDecr = 1
                countSkip = 1

            elif diff == -1:
                countDecr += 1
                countIncr = 1
                countSkip = 1

            elif abs(diff) == 2:
                countSkip += 1
                countIncr = 1
                countDecr = 1

            else:
                countIncr = 1
                countDecr = 1
                countSkip = 1

            maxCount = max(maxCount, countIncr, countDecr, countSkip)

        else:
            countIncr = 1
            countDecr = 1
            countSkip = 1

    if maxCount < 3:
        return 0

    return maxCount / length


# =============================================================================
# LAYER 3: SEMANTIC / DICTIONARY SIMILARITY FEATURES
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "10000_common_passwords.csv")

df = pd.read_csv(DATA_PATH)

commonWords = df.iloc[:, 0].dropna().str.lower().tolist()
commonWords_set = set(commonWords)       # For O(1) exact match lookups
fuzzy_subset = commonWords[:1000]        # Limited fuzzy scan for performance


def maxDictionarySimilarity(password, dictionary):
    """
    Computes a similarity score [0, 1] between the password and the common
    credential dictionary using three tiers:

      1. Hard-coded high-risk pattern check (qwerty, password, admin, etc.)
      2. Exact match in set — O(1)
      3. Substring and token match against full dictionary
      4. Fuzzy partial_ratio match against top-1000 subset (for typo variants)

    Used as the ML feature (dictionaryScore). Produces a continuous score
    for gradient-based learning.

    Note: This is distinct from hasDictionaryPattern(), which is used only
    for binary rule-based penalties. See module docstring for full rationale.
    """
    pwd = password.lower()
    maxScore = 0

    common_patterns = [
        "qwerty", "asdf", "zxcv", "1234", "12345", "123456", "password", "admin", "iloveyou"
    ]
    for pattern in common_patterns:
        if pattern in pwd:
            return 1.0

    if pwd in commonWords_set:
        return 1.0

    tokens = set(re.findall(r'[a-zA-Z]+|\d+', pwd))

    for word in dictionary:
        if len(word) > len(pwd):
            continue

        if word in pwd and len(word) >= 4:
            maxScore = 100

        if word in tokens:
            maxScore = 100

        if maxScore == 100:
            break

    if maxScore < 80:
        for word in fuzzy_subset:
            if abs(len(word) - len(pwd)) > 3:
                continue

            score = fuzz.partial_ratio(pwd, word)
            maxScore = max(maxScore, score)

            if maxScore == 100:
                break

    return maxScore / 100


# Keyboard adjacency graph (QWERTY layout)
KEYBOARD_GRAPH = {
    'q': ['w', 'a'],
    'w': ['q', 'e', 's'],
    'e': ['w', 'r', 'd'],
    'r': ['e', 't', 'f'],
    't': ['r', 'y', 'g'],
    'y': ['t', 'u', 'h'],
    'u': ['y', 'i', 'j'],
    'i': ['u', 'o', 'k'],
    'o': ['i', 'p', 'l'],
    'p': ['o'],

    'a': ['q', 's', 'z'],
    's': ['a', 'd', 'w', 'x'],
    'd': ['s', 'f', 'e', 'c'],
    'f': ['d', 'g', 'r', 'v'],
    'g': ['f', 'h', 't', 'b'],
    'h': ['g', 'j', 'y', 'n'],
    'j': ['h', 'k', 'u', 'm'],
    'k': ['j', 'l', 'i'],
    'l': ['k', 'o'],

    'z': ['a', 'x'],
    'x': ['z', 'c', 's'],
    'c': ['x', 'v', 'd'],
    'v': ['c', 'b', 'f'],
    'b': ['v', 'n', 'g'],
    'n': ['b', 'm', 'h'],
    'm': ['n', 'j'],

    '1': ['2', 'q'],
    '2': ['1', '3', 'w'],
    '3': ['2', '4', 'e'],
    '4': ['3', '5', 'r'],
    '5': ['4', '6', 't'],
    '6': ['5', '7', 'y'],
    '7': ['6', '8', 'u'],
    '8': ['7', '9', 'i'],
    '9': ['8', '0', 'o'],
    '0': ['9', 'p']
}


def keyboardAdjacencyScore(password):
    """
    Detects keyboard adjacency chains (e.g., qwerty, asdfgh, 12345).

    Finds the longest consecutive run of characters that are physically
    adjacent on a QWERTY keyboard layout. Runs shorter than 3 characters
    are ignored to avoid false positives on normal text.

    Applies a boost for chains >= 4 characters to penalize obvious swipe
    patterns more aggressively.
    """
    pwd = password.lower()
    max_chain = 1
    current_chain = 1

    for i in range(1, len(pwd)):
        prev = pwd[i - 1]
        curr = pwd[i]

        if (
            (prev in KEYBOARD_GRAPH and curr in KEYBOARD_GRAPH[prev]) or
            (curr in KEYBOARD_GRAPH and prev in KEYBOARD_GRAPH[curr])
        ):
            current_chain += 1
        else:
            current_chain = 1

        max_chain = max(max_chain, current_chain)

    if len(pwd) == 0:
        return 0

    if max_chain < 3:
        return 0

    score = max_chain / len(pwd)

    if max_chain >= 4:
        score = min(1.0, score * 1.5)
    if max_chain >= 5:
        score = max(score, 0.85)

    return score


# =============================================================================
# BINARY PATTERN DETECTORS (used for rule-based penalties only)
# =============================================================================

def hasDictionaryPattern(password):
    """
    Binary check: returns 1 if the password contains an exact dictionary word
    (length >= 4) as a direct substring. Returns 0 otherwise.

    IMPORTANT: This is intentionally conservative and distinct from
    dictionaryScore (the ML feature). Rule-based penalties are hard overrides
    that can significantly reduce the final score, so they should only trigger
    on confirmed exact matches — not fuzzy near-misses. This prevents
    over-penalizing passwords that merely resemble dictionary words.

    Use dictionaryScore for ML training; use hasDictionaryPattern for penalties.
    """
    pwd = password.lower()
    for word in commonWords_set:
        if word in pwd and len(word) >= 4:
            return 1
    return 0


def hasKeyboardPattern(password):
    """
    Binary check: returns 1 if the password contains a keyboard adjacency chain
    of at least 4 consecutive characters. Returns 0 otherwise.

    Used for rule-based penalty decisions (not as an ML feature).
    """
    pwd = password.lower()
    for i in range(len(pwd) - 3):
        chunk = pwd[i:i + 4]
        if all(
            (chunk[j] in KEYBOARD_GRAPH and chunk[j + 1] in KEYBOARD_GRAPH[chunk[j]])
            for j in range(len(chunk) - 1)
        ):
            return 1
    return 0


def hasSequencePattern(password):
    """
    Binary check: returns 1 if the password contains a sequential character run
    of at least 4 characters (ascending, descending, or skip-2 step).
    Returns 0 otherwise.

    Used for rule-based penalty decisions (not as an ML feature).
    """
    pwd = password.lower()
    for i in range(len(pwd) - 3):
        chunk = pwd[i:i + 4]
        diffs = [ord(chunk[j + 1]) - ord(chunk[j]) for j in range(len(chunk) - 1)]

        if all(d == 1 for d in diffs) or \
           all(d == -1 for d in diffs) or \
           all(abs(d) == 2 for d in diffs):
            return 1
    return 0


# =============================================================================
# MAIN FEATURE EXTRACTION PIPELINE
# =============================================================================

def extractFeatures(password):
    """
    Extracts all 6 normalized features from a password string.

    Returns a dictionary with keys:
        length, maxRepetition, shannonEntropy,
        sequenceCount, dictionaryScore, keyboardScore

    All values are floats in [0, 1].
    """
    return {
        "length":          getLength(password),
        "maxRepetition":   maxRepetitionCount(password),
        "shannonEntropy":  normalize_entropy(password),
        "sequenceCount":   min(1.0, maxPatternSequenceCount(password) * 2.0),
        "dictionaryScore": min(1.0, maxDictionarySimilarity(password, commonWords) * 2.5),
        "keyboardScore":   min(1.0, keyboardAdjacencyScore(password) * 2.0),
    }