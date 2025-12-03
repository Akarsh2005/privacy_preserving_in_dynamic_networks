import os
import glob
import sys
import igraph as ig
from collections import defaultdict
import copy

try:
    import angel as a
except ImportError:
    print("Error: The 'angel_community' library is not installed.")
    print("Please install it by running: pip install angel_community")
    sys.exit(1)


def find_overlapping_nodes(communities):
    """Finds nodes that belong to more than one community."""
    node_counts = defaultdict(int)
    for nodes_in_community in communities.values():
        for node in nodes_in_community:
            node_counts[node] += 1
    return {node for node, count in node_counts.items() if count > 1}


def apply_hiding_strategy(communities, graph):
    """
    Resolves overlaps by assigning each overlapping node to its "best" community.
    The best community is the one containing the most of the node's neighbors.
    """
    initial_overlapping_nodes = find_overlapping_nodes(communities)

    if not initial_overlapping_nodes:
        print("  -> No initial overlapping nodes to hide.")
        return communities

    print(f"  -> Found {len(initial_overlapping_nodes)} overlapping nodes to hide.")

    new_communities = copy.deepcopy(communities)

    # Create a set of all valid vertex names for faster lookups
    valid_node_names = {v["name"] for v in graph.vs}

    for node_id_str in initial_overlapping_nodes:
        # --- ROBUSTNESS CHECK ---
        # Before doing anything, check if the node from the community list
        # actually exists in the graph object. This fixes the error.
        if node_id_str not in valid_node_names:
            print(
                f"    -> Warning: Node '{node_id_str}' found in communities but has no edges in the graph. Removing it."
            )
            # Remove the inconsistent node from all communities it was found in
            for cid in new_communities:
                new_communities[cid].discard(node_id_str)
            continue

        node_id = int(node_id_str)
        neighbors = set(graph.neighbors(node_id_str))

        member_of_comm_ids = [
            cid for cid, nodes in communities.items() if node_id_str in nodes
        ]

        best_community_id = -1
        max_neighbor_count = -1

        # Find the community where the node is most embedded
        for cid in member_of_comm_ids:
            community_nodes = {int(n) for n in communities[cid]}
            neighbor_count = len(neighbors.intersection(community_nodes))

            if neighbor_count >= max_neighbor_count:  # Use >= to handle ties gracefully
                max_neighbor_count = neighbor_count
                best_community_id = cid

        # Now, remove the node from all non-primary communities
        for cid in member_of_comm_ids:
            if cid != best_community_id:
                new_communities[cid].discard(node_id_str)

    return new_communities


def run_hiding_on_snapshots(edge_files_pattern):
    """
    Main function to process each snapshot: detect communities, apply hiding,
    and save the 'after' results.
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

        try:
            # --- 1. Run Initial Community Detection ---
            an = a.Angel(
                network_filename=edge_file_path,
                threshold=0.4,
                min_comsize=3,
                save=False,
                verbose=False,
            )
            initial_communities_dict = an.execute()
            initial_communities = {
                cid: set(nodes) for cid, nodes in initial_communities_dict.items()
            }

            if not initial_communities:
                print("  -> No communities detected initially. Nothing to hide.")
                continue

            print(f"  -> Initially detected {len(initial_communities)} communities.")

            # --- 2. Apply the Hiding Strategy ---
            communities_after_hiding = apply_hiding_strategy(initial_communities, an.G)

            # --- 3. Save the results AFTER HIDING ---
            output_comm_path = os.path.join(
                script_dir,
                filename.replace(".edges", ".angel_communities_after_hiding.txt"),
            )
            print(
                f"  -> Saving new community structure to: {os.path.basename(output_comm_path)}"
            )
            with open(output_comm_path, "w") as f_out:
                for cid, nodes in communities_after_hiding.items():
                    if nodes:
                        sorted_nodes = sorted([int(node) for node in nodes])
                        f_out.write(" ".join(map(str, sorted_nodes)) + "\n")

            final_overlapping_nodes = find_overlapping_nodes(communities_after_hiding)
            output_overlap_path = os.path.join(
                script_dir,
                filename.replace(".edges", ".angel_overlapping_after_hiding.txt"),
            )
            print(
                f"  -> Verifying hiding... Found {len(final_overlapping_nodes)} overlapping nodes remaining."
            )
            print(
                f"  -> Saving new overlapping list to: {os.path.basename(output_overlap_path)}"
            )
            with open(output_overlap_path, "w") as f_out:
                if final_overlapping_nodes:
                    for node in sorted(list(final_overlapping_nodes)):
                        f_out.write(f"{node}\n")

        except Exception as e:
            print(f"  -> An error occurred while processing {filename}: {e}")

    print("\n--- All snapshots have been processed successfully! ---")


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.join(script_directory, "..")
    edge_files_pattern = os.path.join(parent_directory, "mergesplit.t*.edges")

    run_hiding_on_snapshots(edge_files_pattern)
