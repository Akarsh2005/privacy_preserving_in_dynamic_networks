import os
import glob

# --- Configuration ---
# The directory where your 'mergesplit.t*.edges' files are located.
# The 'r' before the string is important to handle Windows backslashes correctly.
input_directory = r"C:\Users\91850\OneDrive\Desktop\pw\akarsh"

# The name of the final combined file that ArchAngel will use.
output_filename = "mergesplit_combined.ncol"
# ---------------------

# Construct the full path for the output file
output_filepath = os.path.join(input_directory, output_filename)

# Create a pattern to find all your snapshot files
file_pattern = os.path.join(input_directory, "mergesplit.t*.edges")

# Use glob to find all files that match the pattern.
# It's CRITICAL to sort the files to ensure they are in chronological order (t01, t02, ..., t10).
snapshot_files = sorted(glob.glob(file_pattern))

if not snapshot_files:
    print(f"Error: No files found matching the pattern: {file_pattern}")
    print("Please make sure your .edges files are in the correct directory.")
else:
    print(
        f"Found {len(snapshot_files)} files to process. Combining them into one file..."
    )

    # Open the single output file in 'write' mode.
    # This will create the file or overwrite it if it already exists.
    with open(output_filepath, "w") as outfile:
        # Loop through each of the found snapshot files
        for filepath in snapshot_files:
            # Get just the filename (e.g., 'mergesplit.t01.edges')
            basename = os.path.basename(filepath)

            # Extract the snapshot ID (e.g., 't01') by splitting the name by '.'
            try:
                snapshot_id = basename.split(".")[-2]
            except IndexError:
                print(f"Could not extract snapshot ID from {basename}. Skipping file.")
                continue

            print(f"  -> Processing {basename} with snapshot ID: {snapshot_id}")

            # Open the current snapshot file to read its contents
            with open(filepath, "r") as infile:
                # Read each line in the file
                for line in infile:
                    # Remove any leading/trailing whitespace and split the line by space
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        node1, node2 = parts[0], parts[1]

                        # Write the formatted line to the output file
                        # The format is: node1<tab>node2<tab>snapshot_id
                        outfile.write(f"{node1}\t{node2}\t{snapshot_id}\n")

    print("\n------------------------------------")
    print("Success!")
    print(f"All snapshots have been combined into: {output_filepath}")
    print("You can now use this file as input for the ArchAngel algorithm.")
    print("------------------------------------")
