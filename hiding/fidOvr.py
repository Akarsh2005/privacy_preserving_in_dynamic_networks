# filename: find_overlapping_nodes.py

def read_communities_from_file(filename):
    """
    Reads community data from a file where each line is in the format:
    <community_id> ['node1', 'node2', ...]
    """
    communities = {}
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            community_id = int(parts[0])
            nodes_str = parts[1].strip()
            # Safely evaluate the list of nodes
            nodes = eval(nodes_str)  # input format uses Python list syntax
            communities[community_id] = nodes
    return communities


def find_overlapping_nodes(communities):
    """
    Returns a dictionary of nodes appearing in more than one community.
    """
    node_to_communities = {}

    for cid, nodes in communities.items():
        for n in nodes:
            node_to_communities.setdefault(n, []).append(cid)

    # filter nodes that appear in multiple communities
    overlapping = {n: cids for n, cids in node_to_communities.items() if len(cids) > 1}
    return overlapping


if __name__ == "__main__":
    filename = "angelArchAngel_coms_2.txt" 
    filename = "angelArchAngel_coms_5.txt"
    communities = read_communities_from_file(filename)
    overlapping = find_overlapping_nodes(communities)

    print("\n✅ Overlapping Nodes:")
    for node, cids in overlapping.items():
        print(f"Node {node} → Communities {cids}")

    print("\nTotal overlapping nodes:", len(overlapping))
