"""
Comprehensive TILES Results Analysis and Comparison
Analyzes edges, communities, overlapping nodes, and comparison metrics
Works from any directory location
"""
import pandas as pd
import json
import gzip
import os
import sys
from collections import defaultdict
import numpy as np

print("="*80)
print("TILES COMPREHENSIVE ANALYSIS AND COMPARISON")
print("="*80)

# Determine correct paths
current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check if we're in TILES subdirectory
if os.path.basename(current_dir) == 'TILES':
    base_dir = os.path.dirname(current_dir)
    tiles_output_dir = 'tiles_output'
else:
    base_dir = script_dir
    tiles_output_dir = 'TILES/tiles_output'

os.chdir(base_dir)
print(f"Working directory: {base_dir}")

# ============================================================================
# PART 1: LOAD GROUND TRUTH DATA
# ============================================================================
print("\n[PART 1] Loading Ground Truth Data")
print("-"*80)

def load_ground_truth():
    """Load all ground truth data"""
    # Communities - read with explicit header
    try:
        communities_df = pd.read_csv('communities.csv', header=0)
    except:
        communities_df = pd.read_csv('communities.csv')
    
    print(f"  Communities CSV shape: {communities_df.shape}")
    print(f"  Communities CSV columns: {list(communities_df.columns)}")
    
    # Normalize column names (remove spaces, lowercase)
    communities_df.columns = [col.strip().lower() for col in communities_df.columns]
    
    gt_communities = {}
    for snapshot in sorted(communities_df['snapshot'].unique()):
        snapshot_data = communities_df[communities_df['snapshot'] == snapshot]
        communities = defaultdict(list)
        
        for _, row in snapshot_data.iterrows():
            communities[int(row['community_id'])].append(str(row['node_id']))
        
        gt_communities[snapshot] = dict(communities)
    
    # Overlapping nodes
    try:
        overlapping_df = pd.read_csv('overlapping_nodes.csv', header=0)
    except:
        overlapping_df = pd.read_csv('overlapping_nodes.csv')
    
    overlapping_df.columns = [col.strip().lower() for col in overlapping_df.columns]
    
    gt_overlapping = {}
    for snapshot in sorted(overlapping_df['snapshot'].unique()):
        nodes = overlapping_df[overlapping_df['snapshot'] == snapshot]['node_id'].astype(str).tolist()
        gt_overlapping[snapshot] = set(nodes)
    
    # Edges
    try:
        edges_df = pd.read_csv('dynamic_network_with_timestamps.csv', header=0)
    except:
        edges_df = pd.read_csv('dynamic_network_with_timestamps.csv')
    
    edges_df.columns = [col.strip().lower() for col in edges_df.columns]
    
    return gt_communities, gt_overlapping, edges_df

try:
    gt_communities, gt_overlapping, edges_df = load_ground_truth()
    
    print(f"✓ Ground Truth Communities: {len(gt_communities)} snapshots")
    print(f"✓ Ground Truth Overlapping Nodes: {len(gt_overlapping)} snapshots")
    print(f"✓ Ground Truth Edges: {len(edges_df)} total edges")
except Exception as e:
    print(f"ERROR loading ground truth: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# PART 2: LOAD TILES RESULTS
# ============================================================================
print("\n[PART 2] Loading TILES Results")
print("-"*80)

def load_tiles_results(output_dir):
    """Load all TILES output files"""
    tiles_communities = {}
    tiles_graphs = {}
    tiles_merged = {}
    tiles_split = {}
    
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory '{output_dir}' not found!")
        return tiles_communities, tiles_graphs, tiles_merged, tiles_split
    
    files = os.listdir(output_dir)
    print(f"Found {len(files)} files in {output_dir}")
    
    # Look for strong-communities files (TILES output format)
    for filename in files:
        if filename.startswith('strong-communities-') and filename.endswith('.gz'):
            try:
                # Extract snapshot number
                snapshot_num = int(filename.replace('strong-communities-', '').replace('.gz', ''))
                
                filepath = os.path.join(output_dir, filename)
                with gzip.open(filepath, 'rt') as f:
                    data = json.load(f)
                    communities = {}
                    for comm_id, nodes in enumerate(data):
                        communities[comm_id] = [str(n) for n in nodes]
                    tiles_communities[snapshot_num + 1] = communities  # +1 to match GT snapshots
                    
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")
        
        # Load graph edges
        elif filename.startswith('graph-') and filename.endswith('.gz'):
            try:
                snapshot_num = int(filename.replace('graph-', '').replace('.gz', ''))
                filepath = os.path.join(output_dir, filename)
                
                with gzip.open(filepath, 'rt') as f:
                    edges = []
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            edges.append((parts[0], parts[1]))
                    tiles_graphs[snapshot_num + 1] = edges
                    
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")
        
        # Load merged communities info
        elif filename.startswith('merging-') and filename.endswith('.gz'):
            try:
                snapshot_num = int(filename.replace('merging-', '').replace('.gz', ''))
                filepath = os.path.join(output_dir, filename)
                with gzip.open(filepath, 'rt') as f:
                    tiles_merged[snapshot_num + 1] = json.load(f)
            except:
                pass
        
        # Load split communities info
        elif filename.startswith('splitting-') and filename.endswith('.gz'):
            try:
                snapshot_num = int(filename.replace('splitting-', '').replace('.gz', ''))
                filepath = os.path.join(output_dir, filename)
                with gzip.open(filepath, 'rt') as f:
                    tiles_split[snapshot_num + 1] = json.load(f)
            except:
                pass
    
    return tiles_communities, tiles_graphs, tiles_merged, tiles_split

tiles_communities, tiles_graphs, tiles_merged, tiles_split = load_tiles_results(tiles_output_dir)

print(f"✓ TILES Communities: {len(tiles_communities)} snapshots")
print(f"✓ TILES Graphs: {len(tiles_graphs)} snapshots")
print(f"✓ TILES Merge Events: {len(tiles_merged)} snapshots")
print(f"✓ TILES Split Events: {len(tiles_split)} snapshots")

# ============================================================================
# PART 3: IDENTIFY OVERLAPPING NODES IN TILES RESULTS
# ============================================================================
print("\n[PART 3] Identifying Overlapping Nodes in TILES Results")
print("-"*80)

def find_overlapping_nodes(communities_dict):
    """Find nodes that belong to multiple communities"""
    node_to_communities = defaultdict(set)
    
    for comm_id, nodes in communities_dict.items():
        for node in nodes:
            node_to_communities[node].add(comm_id)
    
    overlapping = {node for node, comms in node_to_communities.items() if len(comms) > 1}
    return overlapping, node_to_communities

tiles_overlapping = {}
tiles_node_memberships = {}

for snapshot, communities in tiles_communities.items():
    overlapping, memberships = find_overlapping_nodes(communities)
    tiles_overlapping[snapshot] = overlapping
    tiles_node_memberships[snapshot] = memberships

print(f"✓ Identified overlapping nodes in {len(tiles_overlapping)} snapshots")

# ============================================================================
# PART 4: DETAILED SNAPSHOT-BY-SNAPSHOT ANALYSIS
# ============================================================================
print("\n[PART 4] Snapshot-by-Snapshot Analysis")
print("="*80)

snapshot_analysis = []

for snapshot in sorted(set(list(gt_communities.keys()) + list(tiles_communities.keys()))):
    print(f"\n{'='*80}")
    print(f"SNAPSHOT {snapshot}")
    print(f"{'='*80}")
    
    has_gt = snapshot in gt_communities
    has_tiles = snapshot in tiles_communities
    
    # --- Ground Truth Analysis ---
    if has_gt:
        gt_comms = gt_communities[snapshot]
        gt_nodes = set()
        for nodes in gt_comms.values():
            gt_nodes.update(nodes)
        gt_overlap = gt_overlapping.get(snapshot, set())
        
        print(f"\nGround Truth:")
        print(f"  Communities: {len(gt_comms)}")
        print(f"  Total Nodes: {len(gt_nodes)}")
        print(f"  Overlapping Nodes: {len(gt_overlap)}")
        if len(gt_comms) > 0:
            print(f"  Community Sizes: min={min(len(n) for n in gt_comms.values())}, "
                  f"max={max(len(n) for n in gt_comms.values())}, "
                  f"avg={np.mean([len(n) for n in gt_comms.values()]):.1f}")
    else:
        print(f"\nGround Truth: No data")
        gt_comms = {}
        gt_nodes = set()
        gt_overlap = set()
    
    # --- TILES Analysis ---
    if has_tiles:
        tiles_comms = tiles_communities[snapshot]
        tiles_nodes = set()
        for nodes in tiles_comms.values():
            tiles_nodes.update(nodes)
        tiles_overlap = tiles_overlapping.get(snapshot, set())
        
        # Get edges for this snapshot
        tiles_edges_count = len(tiles_graphs.get(snapshot, []))
        
        print(f"\nTILES Results:")
        print(f"  Communities: {len(tiles_comms)}")
        print(f"  Total Nodes: {len(tiles_nodes)}")
        print(f"  Overlapping Nodes: {len(tiles_overlap)}")
        print(f"  Edges: {tiles_edges_count}")
        if len(tiles_comms) > 0:
            print(f"  Community Sizes: min={min(len(n) for n in tiles_comms.values())}, "
                  f"max={max(len(n) for n in tiles_comms.values())}, "
                  f"avg={np.mean([len(n) for n in tiles_comms.values()]):.1f}")
        
        # Show merge/split events
        if snapshot in tiles_merged and tiles_merged[snapshot]:
            print(f"  Merge Events: {len(tiles_merged[snapshot])}")
        if snapshot in tiles_split and tiles_split[snapshot]:
            print(f"  Split Events: {len(tiles_split[snapshot])}")
    else:
        print(f"\nTILES Results: No data")
        tiles_comms = {}
        tiles_nodes = set()
        tiles_overlap = set()
        tiles_edges_count = 0
    
    # --- Comparison Metrics ---
    if has_gt and has_tiles:
        print(f"\n{'-'*80}")
        print("COMPARISON METRICS")
        print(f"{'-'*80}")
        
        # Node coverage
        common_nodes = gt_nodes & tiles_nodes
        coverage = len(common_nodes) / len(gt_nodes) if len(gt_nodes) > 0 else 0
        
        print(f"\nNode Coverage:")
        print(f"  Common Nodes: {len(common_nodes)}/{len(gt_nodes)} ({coverage:.1%})")
        print(f"  GT Only: {len(gt_nodes - tiles_nodes)}")
        print(f"  TILES Only: {len(tiles_nodes - gt_nodes)}")
        
        # Overlapping nodes comparison
        overlap_precision = len(gt_overlap & tiles_overlap) / len(tiles_overlap) if len(tiles_overlap) > 0 else 0
        overlap_recall = len(gt_overlap & tiles_overlap) / len(gt_overlap) if len(gt_overlap) > 0 else 0
        overlap_f1 = 2 * overlap_precision * overlap_recall / (overlap_precision + overlap_recall) if (overlap_precision + overlap_recall) > 0 else 0
        
        print(f"\nOverlapping Nodes Detection:")
        print(f"  GT Overlapping: {len(gt_overlap)}")
        print(f"  TILES Overlapping: {len(tiles_overlap)}")
        print(f"  Correctly Identified: {len(gt_overlap & tiles_overlap)}")
        print(f"  Precision: {overlap_precision:.3f}")
        print(f"  Recall: {overlap_recall:.3f}")
        print(f"  F1-Score: {overlap_f1:.3f}")
        
        # Community-level metrics
        try:
            from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
            
            # Create label arrays for common nodes
            y_true = []
            y_pred = []
            
            for node in common_nodes:
                # Find GT community (use first one if overlapping)
                gt_comm = None
                for comm_id, nodes in gt_comms.items():
                    if node in nodes:
                        gt_comm = comm_id
                        break
                
                # Find TILES community (use first one if overlapping)
                tiles_comm = None
                for comm_id, nodes in tiles_comms.items():
                    if node in nodes:
                        tiles_comm = comm_id
                        break
                
                if gt_comm is not None and tiles_comm is not None:
                    y_true.append(gt_comm)
                    y_pred.append(tiles_comm)
            
            if len(y_true) > 0:
                nmi = normalized_mutual_info_score(y_true, y_pred)
                ari = adjusted_rand_score(y_true, y_pred)
                
                print(f"\nCommunity Quality Metrics:")
                print(f"  NMI (Normalized Mutual Information): {nmi:.4f}")
                print(f"  ARI (Adjusted Rand Index): {ari:.4f}")
            else:
                nmi = None
                ari = None
        except ImportError:
            print(f"\nCommunity Quality Metrics: sklearn not available")
            nmi = None
            ari = None
        
        # Jaccard similarity for overlapping detection
        if len(common_nodes) > 0:
            jaccard_scores = []
            for node in common_nodes:
                gt_comms_for_node = {cid for cid, nodes in gt_comms.items() if node in nodes}
                tiles_comms_for_node = {cid for cid, nodes in tiles_comms.items() if node in nodes}
                
                if len(gt_comms_for_node | tiles_comms_for_node) > 0:
                    jaccard = len(gt_comms_for_node & tiles_comms_for_node) / len(gt_comms_for_node | tiles_comms_for_node)
                    jaccard_scores.append(jaccard)
            
            avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
            print(f"  Avg Jaccard (membership similarity): {avg_jaccard:.4f}")
        else:
            avg_jaccard = None
        
        # Store for summary
        snapshot_analysis.append({
            'snapshot': snapshot,
            'gt_communities': len(gt_comms),
            'tiles_communities': len(tiles_comms),
            'gt_nodes': len(gt_nodes),
            'tiles_nodes': len(tiles_nodes),
            'common_nodes': len(common_nodes),
            'coverage': coverage,
            'gt_overlapping': len(gt_overlap),
            'tiles_overlapping': len(tiles_overlap),
            'overlap_precision': overlap_precision,
            'overlap_recall': overlap_recall,
            'overlap_f1': overlap_f1,
            'nmi': nmi,
            'ari': ari,
            'avg_jaccard': avg_jaccard,
            'edges': tiles_edges_count
        })

# ============================================================================
# PART 5: OVERALL SUMMARY AND STATISTICS
# ============================================================================
print(f"\n\n{'='*80}")
print("OVERALL SUMMARY STATISTICS")
print(f"{'='*80}")

if snapshot_analysis:
    df_summary = pd.DataFrame(snapshot_analysis)
    
    print(f"\nAverage Metrics Across All Snapshots:")
    print(f"{'-'*80}")
    print(f"Node Coverage:              {df_summary['coverage'].mean():.2%}")
    print(f"Overlapping Node Precision: {df_summary['overlap_precision'].mean():.3f}")
    print(f"Overlapping Node Recall:    {df_summary['overlap_recall'].mean():.3f}")
    print(f"Overlapping Node F1-Score:  {df_summary['overlap_f1'].mean():.3f}")
    
    if df_summary['nmi'].notna().any():
        print(f"NMI Score:                  {df_summary['nmi'].mean():.4f}")
    if df_summary['ari'].notna().any():
        print(f"ARI Score:                  {df_summary['ari'].mean():.4f}")
    if df_summary['avg_jaccard'].notna().any():
        print(f"Avg Jaccard Similarity:     {df_summary['avg_jaccard'].mean():.4f}")
    
    print(f"\nCommunity Detection Accuracy:")
    print(f"{'-'*80}")
    print(f"Avg GT Communities per Snapshot:    {df_summary['gt_communities'].mean():.1f}")
    print(f"Avg TILES Communities per Snapshot: {df_summary['tiles_communities'].mean():.1f}")
    print(f"Difference:                         {(df_summary['tiles_communities'].mean() - df_summary['gt_communities'].mean()):.1f}")
    
    print(f"\nOverlapping Node Detection:")
    print(f"{'-'*80}")
    print(f"Avg GT Overlapping per Snapshot:    {df_summary['gt_overlapping'].mean():.1f}")
    print(f"Avg TILES Overlapping per Snapshot: {df_summary['tiles_overlapping'].mean():.1f}")
    
    # Save detailed results
    df_summary.to_csv('detailed_comparison_results.csv', index=False)
    print(f"\n✓ Detailed results saved to: detailed_comparison_results.csv")
    
    # Create a nice summary table
    print(f"\n\nDETAILED COMPARISON TABLE")
    print(f"{'='*80}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_summary.to_string(index=False))

# ============================================================================
# PART 6: VISUALIZE TRENDS
# ============================================================================
print(f"\n\n{'='*80}")
print("GENERATING TREND VISUALIZATIONS")
print(f"{'='*80}")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    if snapshot_analysis:
        df = pd.DataFrame(snapshot_analysis)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TILES vs Ground Truth: Comparison Metrics Over Time', fontsize=16)
        
        # Plot 1: Number of communities
        axes[0, 0].plot(df['snapshot'], df['gt_communities'], 'o-', label='Ground Truth', linewidth=2, markersize=8)
        axes[0, 0].plot(df['snapshot'], df['tiles_communities'], 's-', label='TILES', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Snapshot')
        axes[0, 0].set_ylabel('Number of Communities')
        axes[0, 0].set_title('Communities Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Overlapping nodes
        axes[0, 1].plot(df['snapshot'], df['gt_overlapping'], 'o-', label='Ground Truth', linewidth=2, markersize=8)
        axes[0, 1].plot(df['snapshot'], df['tiles_overlapping'], 's-', label='TILES', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Snapshot')
        axes[0, 1].set_ylabel('Number of Overlapping Nodes')
        axes[0, 1].set_title('Overlapping Nodes Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Quality metrics
        axes[1, 0].plot(df['snapshot'], df['coverage'], 'o-', label='Node Coverage', linewidth=2, markersize=8)
        if df['nmi'].notna().any():
            axes[1, 0].plot(df['snapshot'], df['nmi'], 's-', label='NMI', linewidth=2, markersize=8)
        axes[1, 0].plot(df['snapshot'], df['overlap_f1'], '^-', label='Overlap F1', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Snapshot')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Quality Metrics Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.05])
        
        # Plot 4: Number of nodes
        axes[1, 1].plot(df['snapshot'], df['gt_nodes'], 'o-', label='Ground Truth', linewidth=2, markersize=8)
        axes[1, 1].plot(df['snapshot'], df['tiles_nodes'], 's-', label='TILES', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Snapshot')
        axes[1, 1].set_ylabel('Number of Nodes')
        axes[1, 1].set_title('Active Nodes Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_trends.png', dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: comparison_trends.png")
        
except ImportError:
    print("matplotlib not available - skipping visualizations")
    print("Install with: pip install matplotlib")
except Exception as e:
    print(f"Error creating visualizations: {e}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print("\nGenerated Files:")
print("  1. detailed_comparison_results.csv - Full numerical results")
print("  2. comparison_trends.png - Visual comparison charts")
print(f"\nAll files saved in: {base_dir}")