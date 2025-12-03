"""
Complete TILES Workflow: Data Preparation, Execution, and Comparison
"""
import sys
import os
import pandas as pd
import json
import gzip
from collections import defaultdict
import numpy as np

print("="*70)
print("TILES COMPLETE WORKFLOW")
print("="*70)

# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================
print("\n[STEP 1] Preparing data for TILES...")

# Read your generated data
df = pd.read_csv('dynamic_network_with_timestamps.csv')
print(f"  Loaded {len(df)} edges from dynamic_network_with_timestamps.csv")

# Convert to TILES format (tab-separated, no header)
tiles_data = df[['source', 'target', 'timestamp']].copy()

# Sort by timestamp (CRITICAL requirement for TILES)
tiles_data = tiles_data.sort_values('timestamp')

# Save as TSV without header
tiles_data.to_csv('tiles_input.tsv', sep='\t', header=False, index=False)

time_min = tiles_data['timestamp'].min()
time_max = tiles_data['timestamp'].max()
time_range_days = (time_max - time_min) / 86400

print(f"  Created tiles_input.tsv")
print(f"  Time range: {time_min} to {time_max}")
print(f"  Duration: {time_range_days:.1f} days")

# ============================================================================
# STEP 2: RUN TILES
# ============================================================================
print("\n[STEP 2] Running TILES algorithm...")

# Add TILES directory to path
tiles_path = os.path.join(os.path.dirname(__file__), 'TILES')
sys.path.insert(0, tiles_path)

try:
    import tiles as t
    print("  TILES module loaded successfully")
except ImportError as e:
    print(f"  ERROR: Could not import TILES: {e}")
    print("  Please ensure TILES is in the correct directory")
    sys.exit(1)

# Setup parameters
input_file = "tiles_input.tsv"
output_path = "./tiles_output"
os.makedirs(output_path, exist_ok=True)

print(f"  Parameters:")
print(f"    - Observation window (obs): 1 day")
print(f"    - Time-to-live (ttl): 2 days")
print(f"    - Mode: TILES (Vanilla with implicit decay)")

# Run TILES
tl = t.TILES(filename=input_file, ttl=2, obs=1)
print("\n  Executing TILES (this may take a while)...")
tl.execute(path=output_path)
print("  ✓ TILES execution complete")

# ============================================================================
# STEP 3: LOAD AND COMPARE RESULTS
# ============================================================================
print("\n[STEP 3] Loading and comparing results...")

def load_ground_truth(communities_file):
    """Load ground truth communities by snapshot"""
    df = pd.read_csv(communities_file)
    ground_truth = {}
    
    for snapshot in sorted(df['snapshot'].unique()):
        snapshot_data = df[df['snapshot'] == snapshot]
        communities = defaultdict(list)
        
        for _, row in snapshot_data.iterrows():
            communities[int(row['community_id'])].append(str(row['node_id']))
        
        ground_truth[snapshot] = {k: v for k, v in communities.items()}
    
    return ground_truth

def load_tiles_results(tiles_output_dir):
    """Load TILES detected communities from gzipped JSON files"""
    tiles_results = {}
    
    # List all community files
    files = [f for f in os.listdir(tiles_output_dir) if f.startswith('communities_') and f.endswith('.json.gz')]
    
    for filename in sorted(files):
        # Extract snapshot number from filename
        # Format is usually: communities_X.json.gz
        try:
            snapshot_str = filename.replace('communities_', '').replace('.json.gz', '')
            
            filepath = os.path.join(tiles_output_dir, filename)
            with gzip.open(filepath, 'rt') as f:
                data = json.load(f)
                
                # Convert to our format: dict of {comm_id: [nodes]}
                communities = {}
                for comm_id, nodes in enumerate(data):
                    communities[comm_id] = [str(n) for n in nodes]
                
                # Try to map to snapshot number (TILES uses time-based naming)
                # We'll use sequential numbering based on file order
                snapshot_num = len(tiles_results) + 1
                tiles_results[snapshot_num] = communities
                
        except Exception as e:
            print(f"    Warning: Could not process {filename}: {e}")
    
    return tiles_results

def calculate_metrics(ground_truth_communities, detected_communities):
    """Calculate comparison metrics between ground truth and detected communities"""
    
    # Build node-to-communities mappings
    gt_node_to_comms = defaultdict(set)
    det_node_to_comms = defaultdict(set)
    
    for comm_id, nodes in ground_truth_communities.items():
        for node in nodes:
            gt_node_to_comms[node].add(comm_id)
    
    for comm_id, nodes in detected_communities.items():
        for node in nodes:
            det_node_to_comms[node].add(comm_id)
    
    all_nodes = set(gt_node_to_comms.keys()) | set(det_node_to_comms.keys())
    common_nodes = set(gt_node_to_comms.keys()) & set(det_node_to_comms.keys())
    
    # Calculate Jaccard similarity for overlapping nodes
    if len(common_nodes) > 0:
        jaccard_scores = []
        for node in common_nodes:
            gt_comms = gt_node_to_comms[node]
            det_comms = det_node_to_comms[node]
            
            intersection = len(gt_comms & det_comms)
            union = len(gt_comms | det_comms)
            
            if union > 0:
                jaccard_scores.append(intersection / union)
        
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    else:
        avg_jaccard = 0.0
    
    # Calculate NMI if sklearn is available
    try:
        from sklearn.metrics import normalized_mutual_info_score
        
        # Create single-label assignments (use first community for each node)
        y_true = []
        y_pred = []
        
        for node in common_nodes:
            gt_label = min(gt_node_to_comms[node])  # Use first/smallest community ID
            det_label = min(det_node_to_comms[node])
            y_true.append(gt_label)
            y_pred.append(det_label)
        
        if len(y_true) > 0:
            nmi = normalized_mutual_info_score(y_true, y_pred)
        else:
            nmi = 0.0
    except ImportError:
        nmi = None
    
    return {
        'total_nodes': len(all_nodes),
        'common_nodes': len(common_nodes),
        'coverage': len(common_nodes) / len(all_nodes) if len(all_nodes) > 0 else 0,
        'avg_jaccard': avg_jaccard,
        'nmi': nmi
    }

# Load ground truth
print("  Loading ground truth communities...")
ground_truth = load_ground_truth('communities.csv')
print(f"  ✓ Loaded {len(ground_truth)} ground truth snapshots")

# Load TILES results
print("  Loading TILES results...")
tiles_results = load_tiles_results('tiles_output')
print(f"  ✓ Loaded {len(tiles_results)} TILES result snapshots")

# ============================================================================
# STEP 4: DETAILED COMPARISON
# ============================================================================
print("\n[STEP 4] Comparison Results")
print("="*70)

results_summary = []

for snapshot in sorted(ground_truth.keys()):
    if snapshot in tiles_results:
        gt_comms = ground_truth[snapshot]
        tiles_comms = tiles_results[snapshot]
        
        metrics = calculate_metrics(gt_comms, tiles_comms)
        
        print(f"\nSnapshot {snapshot}:")
        print(f"  Ground Truth: {len(gt_comms)} communities")
        print(f"  TILES Found:  {len(tiles_comms)} communities")
        print(f"  Node Coverage: {metrics['coverage']:.2%} ({metrics['common_nodes']}/{metrics['total_nodes']})")
        print(f"  Avg Jaccard:   {metrics['avg_jaccard']:.4f}")
        if metrics['nmi'] is not None:
            print(f"  NMI Score:     {metrics['nmi']:.4f}")
        
        results_summary.append({
            'snapshot': snapshot,
            'gt_communities': len(gt_comms),
            'tiles_communities': len(tiles_comms),
            'coverage': metrics['coverage'],
            'jaccard': metrics['avg_jaccard'],
            'nmi': metrics['nmi']
        })
    else:
        print(f"\nSnapshot {snapshot}: No TILES results found")

# ============================================================================
# STEP 5: SUMMARY STATISTICS
# ============================================================================
if results_summary:
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    df_summary = pd.DataFrame(results_summary)
    
    print(f"\nAverage Metrics Across All Snapshots:")
    print(f"  Average Node Coverage: {df_summary['coverage'].mean():.2%}")
    print(f"  Average Jaccard Score: {df_summary['jaccard'].mean():.4f}")
    if df_summary['nmi'].notna().any():
        print(f"  Average NMI Score:     {df_summary['nmi'].mean():.4f}")
    
    # Save summary to CSV
    df_summary.to_csv('comparison_results.csv', index=False)
    print(f"\n✓ Detailed results saved to: comparison_results.csv")

print("\n" + "="*70)
print("WORKFLOW COMPLETE")
print("="*70)