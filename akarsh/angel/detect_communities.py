import os
import glob
import sys
from collections import defaultdict

# Try to import the 'angel' library. If it's not found, print an error
# message and exit, as the rest of the script cannot run without it.
try:
    import angel as a
except ImportError:
    print("Error: The 'angel_community' library is not installed.")
    print("Please install it by running: pip install angel_community")
    sys.exit(1)


def find_and_save_overlapping_nodes(communities, base_filename, script_dir):
    """
    Identifies nodes belonging to more than one community and saves them
    to a file, with one node ID per line.

    Args:
        communities (dict): The dictionary of communities from the Angel algorithm.
        base_filename (str): The original filename (e.g., 'mergesplit.t01.edges').
        script_dir (str): The directory where the output file will be saved.
    """
    # Use defaultdict to easily count node occurrences
    node_counts = defaultdict(int)

    # Count how many communities each node appears in
    for nodes_in_community in communities.values():
        for node in nodes_in_community:
            node_counts[node] += 1

    # Create a list of nodes that appear in more than one community
    overlapping_nodes = [int(node) for node, count in node_counts.items() if count > 1]

    # --- Define the output file path for the overlapping nodes ---
    output_filename = base_filename.replace(".edges", ".angel_overlapping.txt")
    output_file_path = os.path.join(script_dir, output_filename)

    if not overlapping_nodes:
        print("  -> No overlapping nodes were found.")
        # Create an empty file to indicate it was processed
        open(output_file_path, "w").close()
        return

    # Sort the nodes numerically for consistent output
    overlapping_nodes.sort()

    print(f"  -> Found {len(overlapping_nodes)} overlapping nodes.")
    print(f"  -> Saving overlapping nodes to: {output_file_path}")

    # Save the sorted list of overlapping nodes, one node per line
    with open(output_file_path, "w") as f_out:
        for node in overlapping_nodes:
            f_out.write(f"{str(node)}\n")


def run_community_detection_on_snapshots(edge_files_pattern):
    """
    Finds snapshot files, runs Angel community detection, saves the
    communities, and then finds and saves the overlapping nodes.
    """
    edge_files = sorted(glob.glob(edge_files_pattern))

    if not edge_files:
        print(f"Error: No edge files found matching pattern '{edge_files_pattern}'")
        return

    print(f"Found {len(edge_files)} snapshot files to process...")

    for edge_file_path in edge_files:
        filename = os.path.basename(edge_file_path)
        print(f"\n--- Processing Snapshot: {filename} ---")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        communities_output_path = os.path.join(
            script_dir, filename.replace(".edges", ".angel_communities.txt")
        )

        try:
            # --- 1. Initialize and Execute Angel Algorithm ---
            an = a.Angel(
                network_filename=edge_file_path,
                threshold=0.4,
                min_comsize=3,
                save=False,
                verbose=False,
            )
            communities = an.execute()

            if not communities:
                print("  -> No communities detected.")
                open(communities_output_path, "w").close()
                find_and_save_overlapping_nodes({}, filename, script_dir)
                continue

            # --- 2. Save the Detected Communities ---
            print(f"  -> Detected {len(communities)} communities.")
            print(f"  -> Saving communities to: {communities_output_path}")
            with open(communities_output_path, "w") as f_out:
                for cid, nodes in communities.items():
                    sorted_nodes = sorted([int(node) for node in nodes])
                    f_out.write(" ".join(map(str, sorted_nodes)) + "\n")

            # --- 3. Find and Save Overlapping Nodes ---
            find_and_save_overlapping_nodes(communities, filename, script_dir)

        except Exception as e:
            print(f"  -> An error occurred while processing {filename}: {e}")

    print("\n--- All snapshots have been processed successfully! ---")


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.join(script_directory, "..")
    edge_files_pattern = os.path.join(parent_directory, "mergesplit.t*.edges")

    run_community_detection_on_snapshots(edge_files_pattern)
