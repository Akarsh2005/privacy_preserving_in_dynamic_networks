import os

def merge_modified_networks_to_ncol(input_dir: str, output_file: str):
    """
    Merge all modified snapshot files into a single .ncol file for ArchAngel
    
    Args:
        input_dir: Directory containing modified network files
        output_file: Output .ncol file path
    """
    print(f"Merging modified networks from: {input_dir}")
    print(f"Output file: {output_file}")
    print("="*70)
    
    merged_edges = []
    snapshot_stats = {}
    
    # Process snapshots t01 to t10
    for i in range(1, 11):
        snapshot_id = f"t{i:02d}"
        snapshot_num = str(i)  # timestamp for ncol format
        
        input_file = os.path.join(input_dir, f"mergesplit.{snapshot_id}.edges_modified")
        
        if not os.path.exists(input_file):
            print(f"⚠ Skipping {snapshot_id}: File not found")
            continue
        
        edge_count = 0
        
        # Read the modified edges file
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    
                    if len(parts) >= 2:
                        node1 = parts[0]
                        node2 = parts[1]
                        # Use snapshot number as timestamp
                        merged_edges.append(f"{node1}\t{node2}\t{snapshot_num}\n")
                        edge_count += 1
        
        snapshot_stats[snapshot_id] = edge_count
        print(f"✓ {snapshot_id}: {edge_count} edges")
    
    # Write merged file
    print(f"\nWriting merged file...")
    with open(output_file, 'w') as f:
        f.writelines(merged_edges)
    
    print(f"✓ Merged file created: {output_file}")
    print(f"Total edges: {len(merged_edges)}")
    print("="*70)
    
    # Print summary
    print("\nSUMMARY:")
    print("-"*70)
    for snapshot_id, count in sorted(snapshot_stats.items()):
        print(f"{snapshot_id}: {count} edges")
    print(f"\nTotal snapshots: {len(snapshot_stats)}")
    print(f"Total edges: {len(merged_edges)}")
    
    return output_file


def merge_original_networks_to_ncol(input_dir: str, output_file: str):
    """
    Merge original snapshot files into a single .ncol file for comparison
    
    Args:
        input_dir: Directory containing original .edges files
        output_file: Output .ncol file path
    """
    print(f"Merging original networks from: {input_dir}")
    print(f"Output file: {output_file}")
    print("="*70)
    
    merged_edges = []
    snapshot_stats = {}
    
    # Process snapshots t01 to t10
    for i in range(1, 11):
        snapshot_id = f"t{i:02d}"
        snapshot_num = str(i)
        
        input_file = os.path.join(input_dir, f"mergesplit.{snapshot_id}.edges")
        
        if not os.path.exists(input_file):
            print(f"⚠ Skipping {snapshot_id}: File not found")
            continue
        
        edge_count = 0
        
        # Read the edges file
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    
                    if len(parts) >= 2:
                        node1 = parts[0]
                        node2 = parts[1]
                        merged_edges.append(f"{node1}\t{node2}\t{snapshot_num}\n")
                        edge_count += 1
        
        snapshot_stats[snapshot_id] = edge_count
        print(f"✓ {snapshot_id}: {edge_count} edges")
    
    # Write merged file
    print(f"\nWriting merged file...")
    with open(output_file, 'w') as f:
        f.writelines(merged_edges)
    
    print(f"✓ Merged file created: {output_file}")
    print(f"Total edges: {len(merged_edges)}")
    print("="*70)
    
    return output_file


if __name__ == "__main__":
    # Merge modified networks (after BIH)
    print("\n" + "#"*70)
    print("MERGING MODIFIED NETWORKS (AFTER BIH)")
    print("#"*70 + "\n")
    
    merge_modified_networks_to_ncol(
        input_dir="BIH_Results/modified_networks",
        output_file="BIH_Results/merged_modified_snapshots.ncol"
    )
    
    print("\n" + "#"*70)
    print("MERGING ORIGINAL NETWORKS (FOR COMPARISON)")
    print("#"*70 + "\n")
    
    # Also create merged original for comparison
    merge_original_networks_to_ncol(
        input_dir="GroundTruth",
        output_file="BIH_Results/merged_original_snapshots.ncol"
    )
    
    print("\n" + "="*70)
    print("READY TO RUN ARCHANGEL!")
    print("="*70)
    print("\nModified network: BIH_Results/merged_modified_snapshots.ncol")
    print("Original network: BIH_Results/merged_original_snapshots.ncol")
    print("\nRun ArchAngel on both files to compare results!")
    print("="*70 + "\n")