#!/usr/bin/env python3
# exam_solution.py
# Plain Python 3 (only psycopg2 is needed for the PostgreSQL part — optional)

from collections import Counter
import re
import random
import math

# -------------------------
# Raw data (copied from HTML)
# -------------------------
RAW = """
MONDAY: GREEN, YELLOW, GREEN, BROWN, BLUE, PINK, BLUE, YELLOW, ORANGE, CREAM, ORANGE, RED, WHITE, BLUE, WHITE, BLUE, BLUE, BLUE, GREEN
TUESDAY: ARSH, BROWN, GREEN, BROWN, BLUE, BLUE, BLEW, PINK, PINK, ORANGE, ORANGE, RED, WHITE, BLUE, WHITE, WHITE, BLUE, BLUE, BLUE
WEDNESDAY: GREEN, YELLOW, GREEN, BROWN, BLUE, PINK, RED, YELLOW, ORANGE, RED, ORANGE, RED, BLUE, BLUE, WHITE, BLUE, BLUE, WHITE, WHITE
THURSDAY: BLUE, BLUE, GREEN, WHITE, BLUE, BROWN, PINK, YELLOW, ORANGE, CREAM, ORANGE, RED, WHITE, BLUE, WHITE, BLUE, BLUE, BLUE, GREEN
FRIDAY: GREEN, WHITE, GREEN, BROWN, BLUE, BLUE, BLACK, WHITE, ORANGE, RED, RED, RED, WHITE, BLUE, WHITE, BLUE, BLUE, BLUE, WHITE
"""

# -------------------------
# Helpers: parse and normalize
# -------------------------
def normalize_color(token: str) -> str:
    t = token.strip().upper()
    if t == "BLEW":    # fix the obvious typo
        return "BLUE"
    # keep unknown tokens as-is (e.g. "ARSH") -- you may want to fix more typos manually
    return t

def parse_raw(raw: str):
    """Return dict day -> [colors], and a flattened list of all colors."""
    days = {}
    all_colors = []
    for line in raw.strip().splitlines():
        if ":" not in line:
            continue
        day, colors_part = line.split(":", 1)
        parts = re.split(r",\s*", colors_part.strip())
        normed = [normalize_color(p) for p in parts if p.strip() != ""]
        days[day.strip().upper()] = normed
        all_colors.extend(normed)
    return days, all_colors

# -------------------------
# Statistics functions
# -------------------------
def color_frequencies(all_colors):
    return Counter(all_colors)

def mode_colors(freq_counter):
    """Return list of color(s) with max frequency and that frequency."""
    if not freq_counter:
        return [], 0
    most = freq_counter.most_common()
    top_count = most[0][1]
    top_colors = [c for c, cnt in most if cnt == top_count]
    return top_colors, top_count

def median_color_by_frequency(freq_counter):
    """Assumption: sort colors by frequency ascending and pick middle color(s)."""
    items = sorted(freq_counter.items(), key=lambda x: x[1])  # ascending
    n = len(items)
    if n == 0:
        return []
    mid = n // 2
    if n % 2 == 1:
        return [items[mid][0]]  # single median color
    else:
        return [items[mid-1][0], items[mid][0]]  # two middle colors

def variance_of_frequencies(freq_counter, population=True):
    """Return variance of the frequency values.
       population=True gives population variance; False gives sample variance."""
    values = list(freq_counter.values())
    if not values:
        return 0.0
    mean = sum(values)/len(values)
    if population:
        var = sum((v-mean)**2 for v in values)/len(values)
    else:
        if len(values) < 2:
            var = 0.0
        else:
            var = sum((v-mean)**2 for v in values)/(len(values)-1)
    return var

def probability_of_color(freq_counter, color):
    total = sum(freq_counter.values())
    if total == 0:
        return 0.0
    return freq_counter.get(color.strip().upper(), 0) / total

# -------------------------
# Save to PostgreSQL
# -------------------------
def save_to_postgres(freq_counter, conn_info):
    """
    Save colors and frequencies to PostgreSQL.
    conn_info: dict with keys: dbname, user, password, host, port
    Requires psycopg2: pip install psycopg2-binary
    """
    try:
        import psycopg2
    except ImportError:
        raise RuntimeError("psycopg2 is required for PostgreSQL. Install with: pip install psycopg2-binary")

    sql_create = """
    CREATE TABLE IF NOT EXISTS color_freq (
        color TEXT PRIMARY KEY,
        freq INTEGER NOT NULL
    );
    """
    sql_upsert = """
    INSERT INTO color_freq (color, freq)
    VALUES (%s, %s)
    ON CONFLICT (color) DO UPDATE SET freq = EXCLUDED.freq;
    """

    conn = psycopg2.connect(**conn_info)
    cur = conn.cursor()
    cur.execute(sql_create)
    for color, freq in freq_counter.items():
        cur.execute(sql_upsert, (color, freq))
    conn.commit()
    cur.close()
    conn.close()

# SQLite fallback (optional, no install required)
def save_to_sqlite(freq_counter, filename="color_freq.sqlite"):
    import sqlite3
    conn = sqlite3.connect(filename)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS color_freq (color TEXT PRIMARY KEY, freq INTEGER NOT NULL)")
    for color, freq in freq_counter.items():
        cur.execute("INSERT OR REPLACE INTO color_freq(color,freq) VALUES (?,?)", (color, freq))
    conn.commit()
    cur.close()
    conn.close()

# -------------------------
# Recursive search algorithms
# -------------------------
def recursive_linear_search(lst, target, index=0):
    """Return index of target or -1. Works on unsorted lists."""
    if index >= len(lst):
        return -1
    if lst[index] == target:
        return index
    return recursive_linear_search(lst, target, index+1)

def recursive_binary_search(sorted_lst, target, left=0, right=None):
    """Recursive binary search: returns index or -1. REQUIRE sorted_lst is sorted."""
    if right is None:
        right = len(sorted_lst) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if sorted_lst[mid] == target:
        return mid
    elif sorted_lst[mid] > target:
        return recursive_binary_search(sorted_lst, target, left, mid-1)
    else:
        return recursive_binary_search(sorted_lst, target, mid+1, right)

# -------------------------
# Random 4 digit binary to decimal
# -------------------------
def random_4bit_binary():
    bits = [str(random.choice([0,1])) for _ in range(4)]
    bstr = "".join(bits)
    decimal = int(bstr, 2)  # convert base-2 string to decimal
    return bstr, decimal

# -------------------------
# Sum of first 50 Fibonacci numbers
# -------------------------
def sum_first_n_fib(n):
    # We'll treat Fibonacci as F0=0, F1=1, F2=1, ...; sum first n numbers means F0..F(n-1)
    a, b = 0, 1
    s = a  # include F0
    for _ in range(1, n):
        s += b
        a, b = b, a+b
    return s

# -------------------------
# Binary sliding-window rule (the exam mapping)
# -------------------------
def sliding_triple_ones(input_bits):
    """For each index i produce '1' if the 3-bit window ending at i is '111', else '0'.
       The output string has same length as input (leading positions will be '0' because
       there's no full 3-bit window until index 2)."""
    out = []
    for i in range(len(input_bits)):
        if i-2 >= 0 and input_bits[i-2:i+1] == "111":
            out.append("1")
        else:
            out.append("0")
    return "".join(out)

# -------------------------
# Put it all together and print results
# -------------------------
def main():
    days, all_colors = parse_raw(RAW)
    freq = color_frequencies(all_colors)

    print("Total color observations:", len(all_colors))
    print("Frequencies:")
    for color, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        print(f"  {color}: {cnt}")

    # Q1 & Q2: mean color / mostly worn
    modes, top_count = mode_colors(freq)
    print("\nMean color (most frequent):", modes, "appeared", top_count, "times")

    # Q3: median color (by frequency ordering)
    med = median_color_by_frequency(freq)
    print("Median color (by frequency order):", med)

    # Q4: variance of counts (population)
    var_pop = variance_of_frequencies(freq, population=True)
    print("Variance of color frequencies (population):", var_pop)

    # Q5: probability color is RED
    prob_red = probability_of_color(freq, "RED")
    print("Probability a randomly chosen color is RED:", prob_red, f"({prob_red*100:.4f}%)")

    # Q6: save to PostgreSQL (example)
    print("\n-- To save to PostgreSQL, call save_to_postgres(freq, conn_info).")
    print("Example conn_info = { 'dbname':'yourdb', 'user':'you', 'password':'pw', 'host':'localhost', 'port':5432 }")
    print("If you don't want to use PostgreSQL, use the sqlite fallback: save_to_sqlite(freq).")

    # Q7: recursive searching demo
    sample_list = [3, 7, 2, 9, 11]
    target = 9
    li_index = recursive_linear_search(sample_list, target)
    print("\nRecursive linear search: list", sample_list, "target", target, "-> index", li_index)
    # For recursive binary search use a sorted list:
    sorted_list = sorted(sample_list)
    bin_index = recursive_binary_search(sorted_list, target)
    print("Recursive binary search (on sorted list)", sorted_list, "target", target, "-> index", bin_index)

    # Q8: random 4-bit binary -> decimal
    bin_str, dec = random_4bit_binary()
    print("\nRandom 4-bit binary:", bin_str, "-> decimal:", dec)

    # Q9: sum first 50 Fibonacci numbers
    sum50 = sum_first_n_fib(50)
    print("Sum of first 50 Fibonacci numbers (F0..F49):", sum50)

    # Sliding window rule example
    inp = "0101101011101011011101101000111"
    out = sliding_triple_ones(inp)
    print("\nSliding-triple-ones input :", inp)
    print("Sliding-triple-ones output:", out)

if __name__ == "__main__":
    main()
