import os
import time

# --- Configuration ---
DATA_DIR = '.' 
OUTPUT_FILE = 'events.tsv'
SECONDS_PER_DAY = 86400  # 24 hours * 60 minutes * 60 seconds
# -------------------

def get_timestamp_from_filename(filename):
    """Extracts the integer timestamp from filenames like 'mergesplit.t01.edges'."""
    try:
        return int(filename.split('.')[1].replace('t', ''))
    except (ValueError, IndexError):
        return -1

def read_edges(filepath):
    """Reads an edge file and returns a set of frozensets for each edge."""
    edges = set()
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            node1, node2 = sorted([parts[0], parts[1]])
            edges.add(frozenset([node1, node2]))
    return edges

# --- Main Script Logic ---
all_files = [f for f in os.listdir(DATA_DIR) if f.startswith('mergesplit') and f.endswith('.edges')]
all_files.sort(key=get_timestamp_from_filename)

if not all_files:
    print("Error: No 'mergesplit.*.edges' files found in the current directory.")
else:
    print(f"Found {len(all_files)} network snapshots to process.")
    previous_edges = set()
    
    # --- THIS IS THE FIX ---
    # Get a base timestamp (e.g., today) to start from.
    base_timestamp = int(time.time())
    # --- END OF FIX ---

    with open(OUTPUT_FILE, 'w') as out_f:
        for filename in all_files:
            step = get_timestamp_from_filename(filename)
            if step == -1:
                print(f"Warning: Skipping file with unrecognized format: {filename}")
                continue
            
            # --- THIS IS THE FIX ---
            # Calculate a new timestamp for each step, one day apart.
            # We use (step - 1) because the first step should be at the base time.
            timestamp = base_timestamp + ((step - 1) * SECONDS_PER_DAY)
            # --- END OF FIX ---

            filepath = os.path.join(DATA_DIR, filename)
            print(f"Processing {filename} with timestamp {timestamp}...")
            current_edges = read_edges(filepath)
            
            added_edges = current_edges - previous_edges
            deleted_edges = previous_edges - current_edges
            
            for edge in added_edges:
                node1, node2 = list(edge)
                out_f.write(f"+\t{node1}\t{node2}\t{timestamp}\n")
                
            for edge in deleted_edges:
                node1, node2 = list(edge)
                out_f.write(f"-\t{node1}\t{node2}\t{timestamp}\n")
                
            previous_edges = current_edges

    print(f"\n✅ Success! Conversion complete. Event stream saved to '{OUTPUT_FILE}'.")