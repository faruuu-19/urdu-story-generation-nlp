"""
Inspect the n_grams.pkl model structure to understand
what's inside and whether a vocabulary mapping exists.
"""
import pickle
import sys
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "backend" / "n_grams.pkl"

if not MODEL_PATH.exists():
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

print(f"Type of loaded data: {type(data).__name__}")
print(f"Total top-level keys/entries: {len(data) if hasattr(data, '__len__') else 'N/A'}")
print()

if isinstance(data, dict):
    # Check for string keys that might be metadata or vocab
    string_keys = [k for k in data.keys() if isinstance(k, str)]
    tuple_keys = [k for k in data.keys() if isinstance(k, tuple)]
    int_keys = [k for k in data.keys() if isinstance(k, int)]
    other_keys = [k for k in data.keys() if not isinstance(k, (str, tuple, int))]
    
    print(f"String keys ({len(string_keys)}): {string_keys[:20]}")
    print(f"Tuple keys ({len(tuple_keys)}): first 5 = {tuple_keys[:5]}")
    print(f"Int keys ({len(int_keys)}): first 10 = {int_keys[:10]}")
    print(f"Other keys ({len(other_keys)}): first 5 = {other_keys[:5]}")
    print()
    
    # Check if any string key holds a dict that could be vocab
    for sk in string_keys:
        val = data[sk]
        print(f"  String key '{sk}': type={type(val).__name__}, value={str(val)[:200]}")
        if isinstance(val, dict):
            sample_items = list(val.items())[:5]
            print(f"    Sample items: {sample_items}")
        elif isinstance(val, list):
            print(f"    Length: {len(val)}, first 5: {val[:5]}")
    
    print()
    
    # Inspect tuple keys
    if tuple_keys:
        first_tuple = tuple_keys[0]
        print(f"First tuple key: {first_tuple}")
        print(f"  Tuple element types: {[type(x).__name__ for x in first_tuple]}")
        first_val = data[first_tuple]
        print(f"  Value type: {type(first_val).__name__}")
        if isinstance(first_val, dict):
            sample_items = list(first_val.items())[:5]
            print(f"  Sample value items: {sample_items}")
            if sample_items:
                ik, iv = sample_items[0]
                print(f"  Inner key type: {type(ik).__name__}, Inner value type: {type(iv).__name__}")
    
    # Inspect int keys
    if int_keys:
        first_int = int_keys[0]
        print(f"\nFirst int key: {first_int}")
        first_val = data[first_int]
        print(f"  Value type: {type(first_val).__name__}")
        print(f"  Value: {str(first_val)[:300]}")
        
        # Check if int keys map to strings (vocab!)
        str_vals = sum(1 for k in int_keys[:100] if isinstance(data[k], str))
        print(f"  Int keys that map to strings (first 100): {str_vals}")
        if str_vals > 0:
            print("  >>> THIS COULD BE THE VOCABULARY (index -> word) <<<")
            sample_vocab = {k: data[k] for k in int_keys[:10]}
            print(f"  Sample vocab: {sample_vocab}")

elif isinstance(data, (list, tuple)):
    print(f"Data is a {type(data).__name__} with {len(data)} elements")
    for i, item in enumerate(data[:5]):
        print(f"  [{i}]: type={type(item).__name__}, value={str(item)[:200]}")
        if isinstance(item, dict):
            print(f"    Keys count: {len(item)}, sample keys: {list(item.keys())[:5]}")

else:
    print(f"Unexpected data type: {type(data).__name__}")
    print(f"Value: {str(data)[:500]}")
