import networkx as nx
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Set, Tuple


class BIHDynamic:
    """
    Based Importance Hiding for Dynamic Networks
    Uses existing ArchAngel community detection results
    Modifies network edges and saves to new files for re-detection
    """
    
    def __init__(self, ground_truth_dir: str, archangel_results_dir: str, 
                 output_dir: str = "BIH_Results"):
        self.ground_truth_dir = ground_truth_dir
        self.archangel_results_dir = archangel_results_dir
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create subdirectories
        self.modified_networks_dir = os.path.join(output_dir, "modified_networks")
        self.reports_dir = os.path.join(output_dir, "reports")
        
        for dir_path in [self.modified_networks_dir, self.reports_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def load_snapshot_network(self, snapshot_file: str) -> nx.Graph:
        """Load a network snapshot from edges format"""
        G = nx.Graph()
        with open(snapshot_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        G.add_edge(parts[0], parts[1])
        return G
    
    def save_graph_to_edges(self, G: nx.Graph, filename: str, snapshot_id: str = "1"):
        """Save graph to edges format (3 columns: node1 node2 timestamp)"""
        with open(filename, 'w') as f:
            for u, v in G.edges():
                f.write(f"{u}\t{v}\t{snapshot_id}\n")
    
    def load_archangel_communities(self, comm_file: str) -> Dict[str, List[str]]:
        """Load communities from ArchAngel output format"""
        communities = {}
        
        if not os.path.exists(comm_file):
            print(f"Warning: Community file not found: {comm_file}")
            return communities
        
        with open(comm_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        comm_id = f"C{parts[0]}"
                        # Parse the node list (removing brackets and quotes)
                        nodes_str = parts[1].strip("[]'\"")
                        nodes = [n.strip("', ") for n in nodes_str.split(',')]
                        nodes = [n.strip() for n in nodes if n.strip()]
                        
                        if len(nodes) >= 2:
                            communities[comm_id] = nodes
        
        return communities
    
    def find_overlapping_nodes(self, communities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Find nodes that belong to multiple communities
        Returns: {node: [list of community IDs]}
        """
        node_to_comms = defaultdict(list)
        
        for comm_id, nodes in communities.items():
            for node in nodes:
                node_to_comms[node].append(comm_id)
        
        # Filter only overlapping nodes (in 2+ communities)
        overlapping = {node: comms for node, comms in node_to_comms.items() 
                      if len(comms) > 1}
        
        return overlapping
    
    def calculate_importance_degree(self, G: nx.Graph, node: str, 
                                   community_nodes: Set[str]) -> float:
        """
        Calculate importance degree of a node in a community
        Formula from paper: I(n, C_i) = (Σ|N_n ∩ N_v|) * (Deg(n)-1) / Deg(n)
        """
        if node not in G:
            return 0.0
        
        degree = G.degree(node)
        if degree == 0 or degree == 1:
            return 0.0
        
        neighbors_n = set(G.neighbors(node)) & community_nodes
        
        if len(neighbors_n) == 0:
            return 0.0
        
        importance_sum = 0
        for v in neighbors_n:
            if v in G:
                neighbors_v = set(G.neighbors(v)) & community_nodes
                common_neighbors = len(neighbors_n & neighbors_v)
                importance_sum += common_neighbors
        
        # Normalized importance
        importance = (importance_sum * (degree - 1)) / degree
        
        return importance
    
    def get_node_with_highest_importance(self, G: nx.Graph, target_node: str,
                                        candidate_nodes: Set[str], 
                                        community_nodes: Set[str]) -> Tuple[str, float]:
        """
        Get the node with highest importance impact for adding/deleting edge
        """
        max_impact = -1
        best_node = None
        
        target_neighbors = set(G.neighbors(target_node)) if target_node in G else set()
        
        for node in candidate_nodes:
            if node == target_node or node not in G:
                continue
            
            # Calculate impact: common neighbors in community
            neighbors_node = set(G.neighbors(node)) & community_nodes
            common = len(target_neighbors & neighbors_node)
            
            # Add the node's degree as tiebreaker
            degree = G.degree(node)
            impact = common * 10 + degree  # Weight common neighbors more
            
            if impact > max_impact:
                max_impact = impact
                best_node = node
        
        return best_node, max_impact
    
    def apply_bih_single_node(self, G: nx.Graph, target_node: str,
                             communities: Dict[str, List[str]],
                             overlapping_comms: List[str],
                             target_comm_id: str,
                             T: int = 5) -> Tuple[nx.Graph, List]:
        """
        Apply BIH algorithm to a single overlapping node
        """
        G_modified = G.copy()
        
        target_comm_nodes = set(communities[target_comm_id])
        other_comms = [c for c in overlapping_comms if c != target_comm_id]
        
        modifications = []
        
        for iteration in range(T):
            # 1. Add edge to target community
            target_neighbors = set(G_modified.neighbors(target_node)) if target_node in G_modified else set()
            non_neighbors = target_comm_nodes - target_neighbors - {target_node}
            
            node_to_add = None
            if non_neighbors:
                # Choose node with highest importance
                node_to_add, impact = self.get_node_with_highest_importance(
                    G_modified, target_node, non_neighbors, target_comm_nodes
                )
                
                if node_to_add and node_to_add in G_modified:
                    G_modified.add_edge(target_node, node_to_add)
                    modifications.append(('add', target_node, node_to_add, target_comm_id, iteration+1))
            
            # 2. Delete edges from other communities
            for other_comm_id in other_comms:
                other_comm_nodes = set(communities[other_comm_id])
                # Recalculate neighbors after potential additions
                target_neighbors = set(G_modified.neighbors(target_node)) if target_node in G_modified else set()
                neighbors_in_comm = target_neighbors & other_comm_nodes
                
                # Keep at least 1 edge to maintain connectivity
                if len(neighbors_in_comm) > 1:
                    # Choose node with highest importance to delete
                    node_to_delete, impact = self.get_node_with_highest_importance(
                        G_modified, target_node, neighbors_in_comm, other_comm_nodes
                    )
                    
                    if node_to_delete and G_modified.has_edge(target_node, node_to_delete):
                        G_modified.remove_edge(target_node, node_to_delete)
                        modifications.append(('delete', target_node, node_to_delete, other_comm_id, iteration+1))
        
        return G_modified, modifications
    
    def process_single_snapshot(self, snapshot_id: str, G: nx.Graph,
                               communities_before: Dict[str, List[str]],
                               T: int = 5, sample_size: int = None):
        """
        Process a single snapshot: apply BIH to overlapping nodes
        """
        print(f"\n{'='*60}")
        print(f"Processing Snapshot: {snapshot_id}")
        print(f"{'='*60}")
        
        print(f"Loaded {len(communities_before)} communities from ArchAngel results")
        
        # Find overlapping nodes
        overlapping_nodes = self.find_overlapping_nodes(communities_before)
        print(f"Found {len(overlapping_nodes)} overlapping nodes")
        
        if not overlapping_nodes:
            print("No overlapping nodes found. Skipping BIH.")
            return None
        
        # Display overlapping nodes
        print("\nOverlapping nodes:")
        for node, comms in list(overlapping_nodes.items())[:10]:  # Show first 10
            print(f"  Node {node}: in {len(comms)} communities {comms}")
        if len(overlapping_nodes) > 10:
            print(f"  ... and {len(overlapping_nodes) - 10} more")
        
        # Sample nodes if requested
        if sample_size and len(overlapping_nodes) > sample_size:
            sampled_nodes = list(overlapping_nodes.keys())[:sample_size]
            overlapping_nodes = {n: overlapping_nodes[n] for n in sampled_nodes}
            print(f"\nProcessing first {sample_size} overlapping nodes")
        
        # Apply BIH to each overlapping node
        results = []
        G_modified = G.copy()
        
        for idx, (node, comm_list) in enumerate(overlapping_nodes.items()):
            print(f"\n[{idx+1}/{len(overlapping_nodes)}] Processing node: {node}")
            print(f"  Currently in {len(comm_list)} communities: {comm_list}")
            
            # Choose random target community
            target_comm = np.random.choice(comm_list)
            print(f"  Target community (to keep): {target_comm}")
            
            # Apply BIH
            G_modified, modifications = self.apply_bih_single_node(
                G_modified, node, communities_before, comm_list, target_comm, T
            )
            
            print(f"  Applied {len(modifications)} edge modifications")
            
            # Count adds vs deletes
            adds = sum(1 for m in modifications if m[0] == 'add')
            deletes = sum(1 for m in modifications if m[0] == 'delete')
            print(f"    - Added: {adds} edges")
            print(f"    - Deleted: {deletes} edges")
            
            results.append({
                'node': node,
                'original_communities': comm_list,
                'target_community': target_comm,
                'modifications': modifications,
                'num_adds': adds,
                'num_deletes': deletes
            })
        
        # Save modified network
        modified_file = os.path.join(
            self.modified_networks_dir, 
            f"mergesplit.{snapshot_id}.edges_modified"
        )
        self.save_graph_to_edges(G_modified, modified_file, snapshot_id.replace('t', ''))
        print(f"\n✓ Modified network saved to: {modified_file}")
        
        # Save detailed report
        self.save_snapshot_report(snapshot_id, results, G, G_modified, communities_before)
        
        return {
            'snapshot_id': snapshot_id,
            'num_overlapping': len(overlapping_nodes),
            'communities_before': communities_before,
            'results': results,
            'graph_original': G,
            'graph_modified': G_modified,
            'modified_file': modified_file
        }
    
    def save_snapshot_report(self, snapshot_id: str, results: List[Dict],
                            G_original: nx.Graph, G_modified: nx.Graph,
                            communities: Dict[str, List[str]]):
        """Save detailed report for a snapshot"""
        report_file = os.path.join(self.reports_dir, f"bih_report_{snapshot_id}.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"BIH HIDING REPORT - {snapshot_id}\n")
            f.write(f"{'='*70}\n\n")
            
            # Network statistics
            f.write("NETWORK STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Original network: {G_original.number_of_nodes()} nodes, "
                   f"{G_original.number_of_edges()} edges\n")
            f.write(f"Modified network: {G_modified.number_of_nodes()} nodes, "
                   f"{G_modified.number_of_edges()} edges\n")
            f.write(f"Edge changes: {G_modified.number_of_edges() - G_original.number_of_edges():+d}\n")
            f.write(f"Communities detected: {len(communities)}\n\n")
            
            # Overall BIH statistics
            total_adds = sum(r['num_adds'] for r in results)
            total_deletes = sum(r['num_deletes'] for r in results)
            
            f.write("BIH MODIFICATIONS SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Nodes processed: {len(results)}\n")
            f.write(f"Total edge additions: {total_adds}\n")
            f.write(f"Total edge deletions: {total_deletes}\n")
            f.write(f"Net edge change: {total_adds - total_deletes:+d}\n\n")
            
            # Per-node details
            f.write("PER-NODE HIDING DETAILS\n")
            f.write("="*70 + "\n\n")
            
            for idx, result in enumerate(results, 1):
                f.write(f"[{idx}] Node: {result['node']}\n")
                f.write("-"*70 + "\n")
                f.write(f"Original communities: {', '.join(result['original_communities'])}\n")
                f.write(f"Target community: {result['target_community']}\n")
                f.write(f"Modifications: {len(result['modifications'])} total "
                       f"({result['num_adds']} adds, {result['num_deletes']} deletes)\n\n")
                
                if result['modifications']:
                    f.write("Modification details:\n")
                    for mod in result['modifications']:
                        action, node1, node2, comm, iteration = mod
                        symbol = "+" if action == "add" else "-"
                        f.write(f"  [{iteration}] {symbol} {action.upper()}: "
                               f"{node1} <-> {node2} (community: {comm})\n")
                
                f.write("\n")
        
        print(f"✓ Report saved to: {report_file}")
    
    def create_instructions_file(self):
        """Create instructions for re-running ArchAngel"""
        instructions_file = os.path.join(self.output_dir, "HOW_TO_RERUN_ARCHANGEL.txt")
        
        with open(instructions_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HOW TO RE-RUN ARCHANGEL ON MODIFIED NETWORKS\n")
            f.write("="*70 + "\n\n")
            
            f.write("The BIH algorithm has modified your networks to hide overlapping nodes.\n")
            f.write("Follow these steps to verify if the hiding was successful:\n\n")
            
            f.write("STEP 1: Prepare the modified network file\n")
            f.write("-"*70 + "\n")
            f.write("Modified network files are in:\n")
            f.write(f"  {self.modified_networks_dir}/\n\n")
            f.write("Each file is named: mergesplit.tXX.edges_modified\n\n")
            
            f.write("STEP 2: Create a combined file (if needed)\n")
            f.write("-"*70 + "\n")
            f.write("If you want to analyze all modified snapshots together:\n")
            f.write("  - Concatenate all mergesplit.tXX.edges_modified files\n")
            f.write("  - OR run ArchAngel on each snapshot individually\n\n")
            
            f.write("STEP 3: Run ArchAngel\n")
            f.write("-"*70 + "\n")
            f.write("Use your existing run_detection.py script, but point to the modified file:\n\n")
            f.write("  import angel as a\n")
            f.write("  \n")
            f.write("  aa = a.ArchAngel(\n")
            f.write("      network_filename='BIH_Results/modified_networks/mergesplit.tXX.edges_modified',\n")
            f.write("      threshold=0.35,\n")
            f.write("      match_threshold=0.35,\n")
            f.write("      min_comsize=3,\n")
            f.write("      save=True,\n")
            f.write("      outfile_path='BIH_Results/archangel_after_bih'\n")
            f.write("  )\n")
            f.write("  aa.execute()\n\n")
            
            f.write("STEP 4: Compare results\n")
            f.write("-"*70 + "\n")
            f.write("Compare the new community files with the original ones:\n")
            f.write("  - Check if hidden nodes are now in single communities\n")
            f.write("  - Calculate ONMI between original and modified structures\n")
            f.write("  - Review the reports in BIH_Results/reports/\n\n")
            
            f.write("STEP 5: Restore original (if needed)\n")
            f.write("-"*70 + "\n")
            f.write("Original networks are untouched in GroundTruth/ directory.\n")
            f.write("Modified networks are separate files, so you can always revert.\n\n")
        
        print(f"\n✓ Instructions saved to: {instructions_file}")
    
    def run_all_snapshots(self, T: int = 5, sample_size: int = None):
        """
        Run BIH on all snapshots using existing ArchAngel results
        """
        print(f"\n{'#'*60}")
        print("BIH DYNAMIC NETWORK HIDING")
        print(f"{'#'*60}\n")
        print("Using existing ArchAngel community detection results")
        print(f"Modifications will be saved to: {self.modified_networks_dir}")
        
        all_results = []
        
        # Process each snapshot
        for i in range(1, 11):  # t01 to t10
            snapshot_id = f"t{i:02d}"
            
            # Load network snapshot
            edge_file = os.path.join(self.ground_truth_dir, f"mergesplit.{snapshot_id}.edges")
            
            # Load ArchAngel communities
            comm_file = os.path.join(self.archangel_results_dir, f"angelArchAngel_coms_{i}.txt")
            
            if not os.path.exists(edge_file):
                print(f"\nSkipping {snapshot_id}: Edge file not found")
                continue
            
            if not os.path.exists(comm_file):
                print(f"\nSkipping {snapshot_id}: ArchAngel results not found")
                print(f"  Expected: {comm_file}")
                continue
            
            # Load graph
            G = self.load_snapshot_network(edge_file)
            communities = self.load_archangel_communities(comm_file)
            
            print(f"\nSnapshot {snapshot_id}:")
            print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"  Communities: {len(communities)}")
            
            # Process snapshot
            result = self.process_single_snapshot(
                snapshot_id, G, communities, T, sample_size
            )
            
            if result:
                all_results.append(result)
        
        # Create instructions file
        self.create_instructions_file()
        
        # Summary
        if all_results:
            self.print_summary(all_results)
        else:
            print("\nNo results to summarize.")
        
        return all_results
    
    def print_summary(self, all_results: List[Dict]):
        """Print summary statistics across all snapshots"""
        print(f"\n{'#'*60}")
        print("SUMMARY ACROSS ALL SNAPSHOTS")
        print(f"{'#'*60}\n")
        
        total_overlapping = sum(r['num_overlapping'] for r in all_results)
        total_processed = sum(len(r['results']) for r in all_results)
        total_adds = sum(sum(nr['num_adds'] for nr in r['results']) for r in all_results)
        total_deletes = sum(sum(nr['num_deletes'] for nr in r['results']) for r in all_results)
        
        print(f"Snapshots processed: {len(all_results)}")
        print(f"Total overlapping nodes found: {total_overlapping}")
        print(f"Total nodes hidden (processed): {total_processed}")
        print(f"\nEdge modifications:")
        print(f"  - Additions: {total_adds}")
        print(f"  - Deletions: {total_deletes}")
        print(f"  - Net change: {total_adds - total_deletes:+d}")
        
        print(f"\n{'='*60}")
        print("NEXT STEPS:")
        print(f"{'='*60}")
        print("1. Review detailed reports in: BIH_Results/reports/")
        print("2. Modified networks are in: BIH_Results/modified_networks/")
        print("3. Run ArchAngel on modified networks to verify hiding")
        print("4. See HOW_TO_RERUN_ARCHANGEL.txt for instructions")
        print(f"{'='*60}\n")


# Main execution
if __name__ == "__main__":
    # Initialize BIH processor
    bih = BIHDynamic(
        ground_truth_dir="GroundTruth",
        archangel_results_dir=".",  # Current directory where angelArchAngel_coms_*.txt files are
        output_dir="BIH_Results"
    )
    
    # Run BIH on all snapshots
    # T = number of iterations (edge rewiring attempts per node)
    # sample_size = number of overlapping nodes to process per snapshot (None = all)
    results = bih.run_all_snapshots(T=5, sample_size=None)
    
    print("\n" + "="*60)
    print("BIH HIDING COMPLETE!")
    print("="*60)