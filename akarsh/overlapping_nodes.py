import os
import glob
from collections import defaultdict

def find_overlapping_nodes_from_comm_files(file_pattern):
    """
    Reads community files, finds nodes that belong to more than one
    community, and saves them to a corresponding '.overlapping.txt' file.
    """
    # Find all files in the current directory matching the pattern
    community_files = sorted(glob.glob(file_pattern))

    if not community_files:
        print(f"Error: No community files found matching the pattern '{file_pattern}'")
        print("Make sure your .comm files are in the same directory as this script.")
        return

    print(f"Found {len(community_files)} community files to process...")

    # Process each file
    for comm_file_path in community_files:
        filename = os.path.basename(comm_file_path)
        print(f"\n--- Processing: {filename} ---")

        # Use a defaultdict to easily count how many times each node appears
        node_community_counts = defaultdict(int)

        try:
            # Read the community file
            with open(comm_file_path, 'r') as f_in:
                for line in f_in:
                    # Split the line into node IDs (they are space-separated)
                    nodes = line.strip().split()
                    # For each node in the community, increment its count
                    for node in nodes:
                        if node: # ensure node string is not empty
                            node_community_counts[node] += 1

            # Identify nodes that appeared in more than 1 community
            overlapping_nodes = [
                int(node) for node, count in node_community_counts.items() if count > 1
            ]

            # Define the output file path
            output_filename = filename.replace(".comm", ".overlapping.txt")
            output_file_path = os.path.join(os.path.dirname(comm_file_path), output_filename)

            if not overlapping_nodes:
                print("  -> No overlapping nodes found in this snapshot.")
                # Create an empty file to show it was processed
                open(output_file_path, 'w').close()
                continue

            # Sort the overlapping nodes numerically for clean, consistent output
            overlapping_nodes.sort()

            print(f"  -> Found {len(overlapping_nodes)} overlapping nodes.")
            print(f"  -> Saving results to: {output_filename}")

            # Write the sorted nodes to the output file, one node per line
            with open(output_file_path, 'w') as f_out:
                for node in overlapping_nodes:
                    f_out.write(f"{node}\n")

        except Exception as e:
            print(f"  -> An error occurred while processing {filename}: {e}")

    print("\n--- All community files have been processed. ---")


if __name__ == "__main__":
    # This script assumes it is in the same directory as the .comm files.
    # It will look for any file ending in '.comm' in this directory.
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_pattern = os.path.join(current_directory, "mergesplit.t*.comm")
    
    find_overlapping_nodes_from_comm_files(file_pattern)