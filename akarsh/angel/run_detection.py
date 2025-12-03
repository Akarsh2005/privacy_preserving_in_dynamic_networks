import os
import sys

# --- This is the key ---
# We are importing the 'ArchAngel' class from the 'angel' package.
try:
    # This imports from the 'angel' folder (angel/__init__.py)
    # which then gets the class from angel/alg/iArchAngel.py
    import angel as a
except ImportError:
    print("--- ERROR ---")
    print("Could not find the 'angel' library.")
    print(
        r"Please make sure the 'angel' folder (from GitHub) is in a location Python can access,"
    )
    print(r"or that you have installed it via pip (`pip install angel_community`).")
    sys.exit(1)

# 1. Define your paths
# The single combined file you created.
# I've updated the filename to match the one from the previous step.
input_file = r"C:\Users\91850\OneDrive\Desktop\pw\akarsh\mergesplit_combined.ncol"

# The directory where you want the results to be saved.
# I've cleaned up the path slightly (removed the extra '\\').
output_path = r"C:\Users\91850\OneDrive\Desktop\pw\akarsh\angel"

# --- Improvement: Check if the input file actually exists ---
if not os.path.exists(input_file):
    print(f"--- ERROR ---")
    print(f"The input file could not be found: {input_file}")
    print("Please make sure you have run the 'create_combined_file.py' script first.")
    sys.exit(1)

# --- Improvement: Ensure the output directory exists ---
if not os.path.exists(output_path):
    print(f"Output directory not found. Creating it now: {output_path}")
    os.makedirs(output_path)

# 2. Initialize ArchAngel
# Your parameters are correct.
aa = a.ArchAngel(
    network_filename=input_file,
    threshold=0.4,  # Merging threshold for Angel
    match_threshold=0.4,  # Matching threshold between snapshots
    min_comsize=3,
    save=True,
    outfile_path=output_path,
)

print(f"Running ArchAngel on: {os.path.basename(input_file)}")
print(f"Saving results to: {output_path}")
print("This may take a few moments...")

# 3. Execute the algorithm
aa.execute()

print("\n------------------------------------")
print("Analysis complete!")
print("ArchAngel has finished processing the dynamic network.")
print(f"Community files and the matches CSV are saved in: {output_path}")
print("------------------------------------")
