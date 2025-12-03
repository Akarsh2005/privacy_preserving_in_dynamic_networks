import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

# ------------------ CONFIGURATION ------------------ #
BASE_DIR = r"C:\Users\91850\OneDrive\Desktop\waste\akarsh"

GROUND_TRUTH_PREFIX = "mergesplit.t"
ARCHANGEL_PREFIX = "angelArchAngel_coms_t"
OUTPUT_DIR = os.path.join(BASE_DIR, "archangel_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------- #


def load_communities(file_path):
    """
    Load community file where each line = list of node IDs in one community.
    Example line: 1 2 3 4
    """
    if not os.path.exists(file_path):
        print(f"⚠️  Warning: File not found {file_path}")
        return None

    communities = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                communities.append(set(parts))
    return communities


def create_node_to_community_map(communities):
    """Map each node → list of community IDs (handles overlapping communities)."""
    mapping = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            mapping.setdefault(node, []).append(cid)
    return mapping


def flatten_mapping_to_labels(mapping):
    """
    Converts overlapping community memberships into single labels for NMI.
    Uses the smallest community ID if overlapping.
    """
    nodes = sorted(mapping.keys())
    labels = [min(mapping[n]) for n in nodes]
    return nodes, labels


def compute_onmi(ground_truth_coms, detected_coms):
    """Compute Normalized Mutual Information (NMI) between two community sets."""
    gt_map = create_node_to_community_map(ground_truth_coms)
    det_map = create_node_to_community_map(detected_coms)

    common_nodes = sorted(set(gt_map.keys()) & set(det_map.keys()))
    if not common_nodes:
        return 0.0

    gt_labels = [min(gt_map[n]) for n in common_nodes]
    det_labels = [min(det_map[n]) for n in common_nodes]

    nmi = normalized_mutual_info_score(gt_labels, det_labels)
    return nmi


# ------------------ MAIN ANALYSIS ------------------ #
def main():
    snapshots = [f"{i:02d}" for i in range(1, 11)]
    onmi_scores = []

    print("\n--- Part 1: ONMI Evaluation ---")

    for snap in snapshots:
        gt_file = os.path.join(BASE_DIR, f"{GROUND_TRUTH_PREFIX}{snap}.comm")
        arch_file = os.path.join(BASE_DIR, f"{ARCHANGEL_PREFIX}{snap}.txt")

        gt_coms = load_communities(gt_file)
        arch_coms = load_communities(arch_file)

        if gt_coms is None or arch_coms is None:
            print(f"⚠️  Missing data for snapshot t{snap}, skipping...\n")
            onmi_scores.append(np.nan)
            continue

        nmi = compute_onmi(gt_coms, arch_coms)
        onmi_scores.append(nmi)
        print(f"✅ Snapshot t{snap}: ONMI = {nmi:.4f}")

    # ------------------ Plotting Results ------------------ #
    valid_scores = [s for s in onmi_scores if not np.isnan(s)]
    if valid_scores:
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

        print(f"\n✅ Plot saved in: {OUTPUT_DIR}\\archangel_onmi_plot.png")
    else:
        print("\n⚠️ No valid ONMI scores computed. Check file paths or formats.")


if __name__ == "__main__":
    main()
