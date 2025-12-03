import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

# ------------------ CONFIGURATION ------------------ #
BASE_DIR = r"C:\Users\91850\OneDrive\Desktop\pw\akarsh"

GROUND_TRUTH_PREFIX = "mergesplit.t"
ARCHANGEL_PREFIX = "angelArchAngel_coms_t"
OUTPUT_DIR = os.path.join(BASE_DIR, "archangel_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------- #


def clean_node_id(node):
    """Normalize node IDs (strip, remove quotes, convert to int if numeric)."""
    node = str(node).strip().replace("'", "").replace('"', "")
    if node.isdigit():
        return str(int(node))  # store as string form of integer
    return node


def load_communities(file_path):
    """
    Load community file where each line = list of node IDs in one community.
    Automatically normalizes all node IDs to strings for consistent comparison.
    """
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: File not found {file_path}")
        return None

    communities = []
    with open(file_path, "r") as f:
        for line in f:
            parts = [clean_node_id(p) for p in line.strip().split() if p.strip()]
            if parts:
                communities.append(set(parts))
    return communities


def load_edges(file_path):
    """Load edges as tuple pairs."""
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: Edge file not found {file_path}")
        return None

    edges = set()
    with open(file_path, "r") as f:
        for line in f:
            parts = [clean_node_id(x) for x in line.strip().split()]
            if len(parts) == 2:
                edges.add(tuple(sorted(parts)))
    return edges


def create_node_to_community_map(communities):
    """Map each node → list of community IDs (handles overlapping communities)."""
    mapping = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            mapping.setdefault(node, []).append(cid)
    return mapping


def compute_onmi(ground_truth_coms, detected_coms):
    """Compute NMI for overlapping communities (simplified)."""
    gt_map = create_node_to_community_map(ground_truth_coms)
    det_map = create_node_to_community_map(detected_coms)

    common_nodes = sorted(set(gt_map.keys()) & set(det_map.keys()))
    if not common_nodes:
        return 0.0

    gt_labels = [min(gt_map[n]) for n in common_nodes]
    det_labels = [min(det_map[n]) for n in common_nodes]

    return normalized_mutual_info_score(gt_labels, det_labels)


def analyze_differences(gt_coms, det_coms):
    """Compare ground truth vs detected communities (node & community overlap)."""
    gt_nodes = set().union(*gt_coms)
    det_nodes = set().union(*det_coms)

    missing_nodes = gt_nodes - det_nodes
    extra_nodes = det_nodes - gt_nodes
    overlap_nodes = gt_nodes & det_nodes

    num_missing_nodes = len(missing_nodes)
    num_extra_nodes = len(extra_nodes)
    overlap_count = len(overlap_nodes)

    # Community-level comparison
    num_gt_coms = len(gt_coms)
    num_det_coms = len(det_coms)
    num_missing_coms = max(0, num_gt_coms - num_det_coms)
    num_extra_coms = max(0, num_det_coms - num_gt_coms)

    # Coverage %
    node_coverage = (overlap_count / len(gt_nodes)) * 100 if gt_nodes else 0

    return (
        num_missing_coms,
        num_extra_coms,
        num_missing_nodes,
        num_extra_nodes,
        node_coverage,
        overlap_count,
        len(gt_nodes),
        len(det_nodes),
    )


# ------------------ MAIN ANALYSIS ------------------ #
def main():
    snapshots = [f"{i:02d}" for i in range(1, 11)]
    onmi_scores = []
    comm_diffs = []
    node_diffs = []

    print("\n--- ArchAngel vs Ground Truth Analysis ---\n")

    for snap in snapshots:
        gt_file = os.path.join(BASE_DIR, f"{GROUND_TRUTH_PREFIX}{snap}.comm")
        arch_file = os.path.join(BASE_DIR, f"{ARCHANGEL_PREFIX}{snap}.txt")
        edge_file = os.path.join(BASE_DIR, f"{GROUND_TRUTH_PREFIX}{snap}.edges")

        gt_coms = load_communities(gt_file)
        arch_coms = load_communities(arch_file)
        gt_edges = load_edges(edge_file)

        if gt_coms is None or arch_coms is None:
            print(f"⚠️  Missing data for snapshot t{snap}, skipping...\n")
            onmi_scores.append(np.nan)
            comm_diffs.append((np.nan, np.nan))
            node_diffs.append((np.nan, np.nan))
            continue

        nmi = compute_onmi(gt_coms, arch_coms)
        onmi_scores.append(nmi)

        (
            miss_coms,
            extra_coms,
            miss_nodes,
            extra_nodes,
            coverage,
            overlap_nodes,
            total_gt_nodes,
            total_arch_nodes,
        ) = analyze_differences(gt_coms, arch_coms)

        comm_diffs.append((miss_coms, extra_coms))
        node_diffs.append((miss_nodes, extra_nodes))

        total_edges = len(gt_edges) if gt_edges else 0

        print(f"📅 Snapshot t{snap}")
        print(f"   ➤ ONMI Score           : {nmi:.4f}")
        print(
            f"   ➤ Ground Truth Comms   : {len(gt_coms)} | ArchAngel Comms : {len(arch_coms)}"
        )
        print(
            f"   ➤ Missing Communities  : {miss_coms} | Extra Communities : {extra_coms}"
        )
        print(
            f"   ➤ Ground Truth Nodes   : {total_gt_nodes} | ArchAngel Nodes : {total_arch_nodes}"
        )
        print(f"   ➤ Missing Nodes        : {miss_nodes} | Extra Nodes : {extra_nodes}")
        print(
            f"   ➤ Overlapping Nodes    : {overlap_nodes} / {total_gt_nodes} ({coverage:.2f}%)"
        )
        print(f"   ➤ Edges in Ground Truth: {total_edges}\n")

    # ------------------ Plot ONMI ------------------ #
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), onmi_scores, marker="o", linewidth=2)
    plt.title("ArchAngel Performance over Time (ONMI)")
    plt.xlabel("Snapshot (t)")
    plt.ylabel("ONMI Score")
    plt.grid(True)
    plt.xticks(range(1, 11))
    plt.ylim(0, 1)
    plt.savefig(os.path.join(OUTPUT_DIR, "archangel_onmi_plot.png"))
    plt.show()

    # ------------------ Summary ------------------ #
    print("\n--- Summary of Missing Data ---")
    print(
        f"{'Snapshot':<10}{'ONMI':<10}{'MissCom':<10}{'ExtraCom':<10}{'MissNode':<10}{'ExtraNode':<10}{'Coverage%':<10}"
    )
    print("-" * 80)

    for i, snap in enumerate(snapshots):
        mc, ec = comm_diffs[i]
        mn, en = node_diffs[i]
        score = onmi_scores[i]
        if np.isnan(score):
            print(f"t{snap:<8}{'NaN':<10}{'-':<10}{'-':<10}{'-':<10}{'-':<10}{'-':<10}")
        else:
            coverage = analyze_differences(
                load_communities(
                    os.path.join(BASE_DIR, f"{GROUND_TRUTH_PREFIX}{snap}.comm")
                ),
                load_communities(
                    os.path.join(BASE_DIR, f"{ARCHANGEL_PREFIX}{snap}.txt")
                ),
            )[4]
            print(
                f"t{snap:<8}{score:<10.4f}{mc:<10}{ec:<10}{mn:<10}{en:<10}{coverage:<10.2f}"
            )


if __name__ == "__main__":
    main()
