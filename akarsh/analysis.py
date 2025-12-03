import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from tabulate import tabulate  # for pretty tables

# ------------------ CONFIGURATION ------------------ #
BASE_DIR = r"C:\Users\91850\OneDrive\Desktop\pw\akarsh"

GROUND_TRUTH_PREFIX = "mergesplit.t"
ARCHANGEL_PREFIX = "angelArchAngel_coms_t"
OUTPUT_DIR = os.path.join(BASE_DIR, "archangel_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------- #


def load_communities(file_path):
    """
    Load community file where each line = list of node IDs in one community.
    Supports both ArchAngel format (tab-separated with [list]) and
    ground truth format (space-separated node IDs).
    """
    if not os.path.exists(file_path):
        print(f"⚠  Warning: File not found {file_path}")
        return None

    communities = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # --- ArchAngel format ---
            if "\t" in line and "[" in line and "]" in line:
                try:
                    idx, nodes_str = line.split("\t", 1)
                    nodes = [
                        n.strip().strip("'").strip('"')
                        for n in nodes_str.strip("[]").split(",")
                        if n.strip().strip("'").strip('"')
                    ]
                    communities.append(set(nodes))
                except Exception as e:
                    print(f"⚠  Error parsing ArchAngel line: {line} ({e})")

            # --- Ground Truth format ---
            else:
                parts = [p.strip().strip("'").strip('"') for p in line.split() if p.strip()]
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


def analyze_communities(ground_truth_coms, detected_coms, threshold=0.3):
    """
    Analyze differences between ground truth and detected communities.
    threshold: Jaccard similarity below which a community is considered "missing".
    """
    gt_nodes = set().union(*ground_truth_coms) if ground_truth_coms else set()
    det_nodes = set().union(*detected_coms) if detected_coms else set()

    # --- Node level comparison ---
    missing_nodes = gt_nodes - det_nodes
    new_nodes = det_nodes - gt_nodes

    # --- Community level comparison using Jaccard similarity ---
    missing_communities = []
    for gt_comm in ground_truth_coms:
        best_overlap = 0
        for det_comm in detected_coms:
            inter = len(gt_comm & det_comm)
            union = len(gt_comm | det_comm)
            if union > 0:
                jaccard = inter / union
                best_overlap = max(best_overlap, jaccard)
        if best_overlap < threshold:
            missing_communities.append(gt_comm)

    n_gt_communities = len(ground_truth_coms)
    n_det_communities = len(detected_coms)

    gt_sizes = [len(comm) for comm in ground_truth_coms]
    det_sizes = [len(comm) for comm in detected_coms]

    avg_gt_size = np.mean(gt_sizes) if gt_sizes else 0
    avg_det_size = np.mean(det_sizes) if det_sizes else 0

    return {
        "missing_nodes": len(missing_nodes),
        "new_nodes": len(new_nodes),
        "gt_communities": n_gt_communities,
        "det_communities": n_det_communities,
        "avg_gt_size": avg_gt_size,
        "avg_det_size": avg_det_size,
        "missing_nodes_list": sorted(list(missing_nodes)),
        "new_nodes_list": sorted(list(new_nodes)),
        "missing_communities": len(missing_communities),
    }


# ------------------ MAIN ANALYSIS ------------------ #
def main():
    snapshots = [f"{i:02d}" for i in range(1, 11)]
    onmi_scores = []
    summary_table = []

    print("\nONMI Evaluation")

    for snap in snapshots:
        gt_file = os.path.join(BASE_DIR, f"{GROUND_TRUTH_PREFIX}{snap}.comm")
        arch_file = os.path.join(BASE_DIR, f"{ARCHANGEL_PREFIX}{snap}.txt")

        gt_coms = load_communities(gt_file)
        arch_coms = load_communities(arch_file)

        if gt_coms is None or arch_coms is None:
            print(f"⚠ Missing data for snapshot t{snap}, skipping...\n")
            onmi_scores.append(np.nan)
            continue

        nmi = compute_onmi(gt_coms, arch_coms)
        onmi_scores.append(nmi)
        print(f"Snapshot t{snap}: ONMI = {nmi:.4f}")

        analysis = analyze_communities(gt_coms, arch_coms)

        summary_table.append(
            [
                f"t{snap}",
                f"{nmi:.4f}",
                analysis["gt_communities"],
                analysis["det_communities"],
                analysis["missing_communities"],
                analysis["missing_nodes"],
                analysis["new_nodes"],
            ]
        )

    # -------------- Display Summary in Terminal --------------
    print("\n================= Community Analysis Summary =================")
    headers = [
        "Snapshot",
        "ONMI Score",
        "GT Communities",
        "Detected Communities",
        "Missing Communities",
        "Missing Nodes",
        "New Nodes",
    ]
    print(tabulate(summary_table, headers=headers, tablefmt="grid"))

    # -------------- Plot Results --------------
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
        print("\n⚠ No valid ONMI scores computed. Check file paths or formats.")


if __name__ == "__main__":
    main()
