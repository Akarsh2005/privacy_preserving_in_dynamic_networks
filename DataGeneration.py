import networkx as nx
import random
import csv
import pandas as pd
from collections import defaultdict

# --- STAGE 1: DATA GENERATION ---

def generate_lfr_style_network(nodes_to_use, mu, avg_degree, min_comm, max_comm, overlap_percentage):
    """
    Generates a network graph (G), its community structure (communities),
    and a list of nodes designated as overlapping.
    """
    G = nx.Graph()
    G.add_nodes_from(nodes_to_use)
    n = len(nodes_to_use)
    
    # 1. Create base communities
    communities = {}
    nodes = list(nodes_to_use)
    random.shuffle(nodes)
    community_id_counter = 0
    while nodes:
        comm_size = random.randint(min_comm, max_comm)
        if not nodes:
            break
        community = nodes[:min(comm_size, len(nodes))]
        communities[community_id_counter] = community
        nodes = nodes[min(comm_size, len(nodes)):]
        community_id_counter += 1

    # 2. Explicitly create overlapping nodes
    all_nodes_flat = [node for comm in communities.values() for node in comm]
    num_overlap = int(len(all_nodes_flat) * overlap_percentage)
    nodes_to_make_overlapping = random.sample(all_nodes_flat, min(num_overlap, len(all_nodes_flat)))
    overlapping_nodes_list = list(nodes_to_make_overlapping)

    for node in overlapping_nodes_list:
        current_comm_id = None
        for cid, members in communities.items():
            if node in members:
                current_comm_id = cid
                break
        
        possible_new_communities = [cid for cid in communities.keys() if cid != current_comm_id]
        if possible_new_communities:
            new_comm_id = random.choice(possible_new_communities)
            communities[new_comm_id].append(node)

    # 3. Add edges based on community structure - SPARSE within communities
    for cid, members in communities.items():
        # Only connect some nodes within each community to create sparse structure
        num_internal_edges = len(members)  # Approximately one edge per node
        for _ in range(num_internal_edges):
            if len(members) < 2:
                break
            i, j = random.sample(range(len(members)), 2)
            if not G.has_edge(members[i], members[j]):
                G.add_edge(members[i], members[j])

    # Very minimal inter-community edges - "bridges" between galaxies
    num_communities = len(communities)
    bridge_edges = max(1, num_communities // 3)  # Few bridges
    
    for _ in range(bridge_edges):
        if len(communities) < 2:
            break
        cid1, cid2 = random.sample(list(communities.keys()), 2)
        if communities[cid1] and communities[cid2]:
            node1 = random.choice(communities[cid1])
            node2 = random.choice(communities[cid2])
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2)
            
    return G, communities, overlapping_nodes_list

def apply_community_dynamics(prev_communities, prev_nodes, current_nodes, snapshot_num, community_id_offset):
    """
    Apply dynamic community changes: merging, splitting, migration, new communities
    """
    new_communities = {}
    next_comm_id = community_id_offset
    
    # Nodes that left and new nodes
    leaving_nodes = set(prev_nodes) - set(current_nodes)
    new_nodes = set(current_nodes) - set(prev_nodes)
    continuing_nodes = set(prev_nodes) & set(current_nodes)
    
    # Filter out leaving nodes from previous communities
    active_communities = {}
    for cid, members in prev_communities.items():
        active_members = [m for m in members if m in continuing_nodes]
        if active_members:
            active_communities[cid] = active_members
    
    # Apply dynamic events based on snapshot
    if snapshot_num == 3:  # Community merge
        if len(active_communities) >= 2:
            comm_ids = list(active_communities.keys())
            merge_id1, merge_id2 = random.sample(comm_ids, 2)
            merged_members = active_communities[merge_id1] + active_communities[merge_id2]
            new_communities[next_comm_id] = merged_members
            next_comm_id += 1
            for cid, members in active_communities.items():
                if cid not in [merge_id1, merge_id2]:
                    new_communities[next_comm_id] = members
                    next_comm_id += 1
        else:
            for members in active_communities.values():
                new_communities[next_comm_id] = members
                next_comm_id += 1
    
    elif snapshot_num == 5:  # Community split
        if active_communities:
            comm_ids = list(active_communities.keys())
            split_cid = random.choice(comm_ids)
            split_members = active_communities[split_cid]
            if len(split_members) >= 4:
                random.shuffle(split_members)
                mid = len(split_members) // 2
                new_communities[next_comm_id] = split_members[:mid]
                next_comm_id += 1
                new_communities[next_comm_id] = split_members[mid:]
                next_comm_id += 1
                for cid, members in active_communities.items():
                    if cid != split_cid:
                        new_communities[next_comm_id] = members
                        next_comm_id += 1
            else:
                for members in active_communities.values():
                    new_communities[next_comm_id] = members
                    next_comm_id += 1
        else:
            for members in active_communities.values():
                new_communities[next_comm_id] = members
                next_comm_id += 1
    
    elif snapshot_num == 7:  # Node migration between communities
        comm_ids = list(active_communities.keys())
        if len(comm_ids) >= 2:
            source_cid = random.choice(comm_ids)
            target_cid = random.choice([c for c in comm_ids if c != source_cid])
            source_members = active_communities[source_cid]
            if len(source_members) > 2:
                migrating_node = random.choice(source_members)
                for cid, members in active_communities.items():
                    if cid == source_cid:
                        new_communities[next_comm_id] = [m for m in members if m != migrating_node]
                    elif cid == target_cid:
                        new_communities[next_comm_id] = members + [migrating_node]
                    else:
                        new_communities[next_comm_id] = members
                    next_comm_id += 1
            else:
                for members in active_communities.values():
                    new_communities[next_comm_id] = members
                    next_comm_id += 1
        else:
            for members in active_communities.values():
                new_communities[next_comm_id] = members
                next_comm_id += 1
    
    else:  # Normal evolution - keep existing communities
        for members in active_communities.values():
            new_communities[next_comm_id] = members
            next_comm_id += 1
    
    # Add new nodes - form new community or join existing ones
    if new_nodes:
        new_nodes_list = list(new_nodes)
        if len(new_nodes_list) >= 8:  # Form new community
            new_communities[next_comm_id] = new_nodes_list
            next_comm_id += 1
        else:  # Join existing communities
            for node in new_nodes_list:
                if new_communities:
                    random_comm = random.choice(list(new_communities.keys()))
                    new_communities[random_comm].append(node)
                else:
                    new_communities[next_comm_id] = [node]
                    next_comm_id += 1
    
    return new_communities, next_comm_id

# --- Main Generation Logic ---
random.seed(42)
all_edges_rows = [['source', 'target', 'timestamp']]
all_communities_rows = [['snapshot', 'community_id', 'node_id']]
all_overlapping_rows = [['snapshot', 'node_id']]

# Node pool management
node_pool = list(range(300))  # Pool of all possible nodes
active_nodes = []
prev_communities = {}
community_id_offset = 0

# Define node count progression for 10 snapshots
node_counts = [120, 140, 165, 155, 190, 210, 240, 235, 220, 200]

# Unix timestamp base (Jan 1, 2020 00:00:00 UTC)
base_unix_time = 1577836800
time_interval = 86400  # 1 day in seconds (24 * 60 * 60)

for i in range(1, 11):  # 10 snapshots
    print(f"Generating Snapshot {i}...")
    
    target_nodes = node_counts[i-1]
    
    # Determine which nodes to keep, remove, and add
    if i == 1:
        active_nodes = random.sample(node_pool, target_nodes)
    else:
        nodes_to_keep = target_nodes - max(0, target_nodes - len(active_nodes))
        if target_nodes < len(active_nodes):
            # Reduce nodes
            active_nodes = random.sample(active_nodes, target_nodes)
        else:
            # Add new nodes
            available_nodes = [n for n in node_pool if n not in active_nodes]
            nodes_to_add = target_nodes - len(active_nodes)
            new_nodes = random.sample(available_nodes, min(nodes_to_add, len(available_nodes)))
            # Some existing nodes might leave
            nodes_leaving = random.randint(0, min(10, len(active_nodes) // 10))
            active_nodes = random.sample(active_nodes, len(active_nodes) - nodes_leaving)
            active_nodes.extend(new_nodes)
    
    # Apply community dynamics if not first snapshot
    if i == 1:
        G, coms, overlapping_nodes = generate_lfr_style_network(
            nodes_to_use=active_nodes, mu=0.2, avg_degree=2.5,  # Lower density
            min_comm=8, max_comm=25, overlap_percentage=0.10  # Smaller, more communities
        )
        prev_communities = coms
        # Renumber communities with offset
        coms = {community_id_offset + cid: members for cid, members in coms.items()}
        community_id_offset += len(coms)
    else:
        coms, community_id_offset = apply_community_dynamics(
            prev_communities, prev_active_nodes, active_nodes, i, community_id_offset
        )
        G, _, overlapping_nodes = generate_lfr_style_network(
            nodes_to_use=active_nodes, mu=0.2, avg_degree=2.5,  # Lower density
            min_comm=8, max_comm=25, overlap_percentage=0.10  # Smaller, more communities
        )
        # Override with dynamic communities
        prev_communities = coms
    
    prev_active_nodes = active_nodes.copy()
    
    # Calculate Unix timestamp range for this snapshot
    time_start = base_unix_time + (i - 1) * time_interval
    time_end = base_unix_time + i * time_interval
    
    for u, v in G.edges():
        all_edges_rows.append([u, v, random.randint(time_start, time_end)])
    
    for cid, members in coms.items():
        for node in members:
            all_communities_rows.append([i, cid, node])
    
    for node in overlapping_nodes:
        all_overlapping_rows.append([i, node])

# --- Save all three files ---
print("\nSaving generated files...")
with open("dynamic_network_with_timestamps.csv", 'w', newline='') as f:
    csv.writer(f).writerows(all_edges_rows)
print("-> dynamic_network_with_timestamps.csv created.")
with open("communities.csv", 'w', newline='') as f:
    csv.writer(f).writerows(all_communities_rows)
print("-> communities.csv created.")
with open("overlapping_nodes.csv", 'w', newline='') as f:
    csv.writer(f).writerows(all_overlapping_rows)
print("-> overlapping_nodes.csv created.")
print("\nGeneration complete!")


# --- STAGE 2: GEXF TRANSFORMATION AND CREATION ---

def create_dynamic_gexf(edge_file, community_file, output_file):
    """
    Combines a temporal edge list and a dynamic community membership file
    into a single dynamic GEXF file for visualization in Gephi.
    """
    print("\nStarting GEXF file creation...")
    edges_df = pd.read_csv(edge_file, dtype={'source': str, 'target': str, 'timestamp': int})
    communities_df = pd.read_csv(community_file)

    # --- Step 1: Transform snapshot community data into spell format ---
    print("Transforming community data to time-interval format...")
    communities_df['start_time'] = (communities_df['snapshot'] - 1) * 10 + 1
    communities_df['end_time'] = communities_df['snapshot'] * 10
    communities_df.rename(columns={'node_id': 'node'}, inplace=True)
    communities_df = communities_df.astype({'node': str, 'community_id': str})

    print("Initializing graph...")
    G = nx.Graph()
    
    # Color palette for communities
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
              '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
              '#e5c494', '#b3b3b3', '#1b9e77', '#d95f02', '#7570b3']

    # --- Step 2: Process nodes and their dynamic community attributes ---
    node_community_spells = defaultdict(list)
    for _, row in communities_df.iterrows():
        comm_id = int(row['community_id'])
        color = colors[comm_id % len(colors)]
        spell = {'value': row['community_id'], 'start': row['start_time'], 'end': row['end_time']}
        node_community_spells[row['node']].append(spell)

    for node, spells in node_community_spells.items():
        # Get the first community for static coloring
        first_comm_id = int(spells[0]['value'])
        node_color = colors[first_comm_id % len(colors)]
        
        G.add_node(node, 
                   community=spells,
                   viz={'color': {'r': int(node_color[1:3], 16), 
                                  'g': int(node_color[3:5], 16), 
                                  'b': int(node_color[5:7], 16)}})
        node_start = min(s['start'] for s in spells)
        node_end = max(s['end'] for s in spells)
        G.nodes[node]['start'] = node_start
        G.nodes[node]['end'] = node_end

    # --- Step 3: Process edges and their temporal spells ---
    for _, row in edges_df.iterrows():
        G.add_edge(str(row['source']), str(row['target']), start=row['timestamp'])
        
    print(f"Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Step 4: Write the dynamic GEXF file ---
    print(f"Writing dynamic GEXF file to {output_file}...")
    nx.write_gexf(G, output_file, version='1.2draft')
    print("Process complete.")

# --- Execute the GEXF creation function ---
create_dynamic_gexf('dynamic_network_with_timestamps.csv', 'communities.csv', 'ground_truth_dynamic.gexf')