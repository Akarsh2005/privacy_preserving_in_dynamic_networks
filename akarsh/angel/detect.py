import time
import sys
import os
import glob
import igraph
import tqdm
from collections import Counter
from six import iteritems  # For Py2/3 compatibility, as in original code

# --- ASSUMPTION ---
# This script assumes a file named 'angel.py' (or 'angel.so', etc.)
# exists in the same directory (C:\...akarsh\angel\),
# from which we can import the 'Angel' class.
try:
    import angel as an
except ImportError:
    print("Error: Could not import 'angel'.")
    print("Please make sure 'angel.py' is in the same directory as this script.")
    sys.exit(1)


# Decorator (as provided)
def timeit(method):
    """
    Decorator: Compute the execution time of a function
    :param method: the function
    :return: the method runtime
    """

    def timed(*arguments, **kw):
        ts = time.time()
        result = method(*arguments, **kw)
        te = time.time()

        sys.stdout.write(
            "Time:  %r %2.2f sec\n" % (method.__name__.strip("_"), te - ts)
        )
        sys.stdout.write("------------------------------------\n")
        sys.stdout.flush()
        return result

    return timed


class ArchAngel(object):

    def __init__(
        self,
        network_files_list,
        output_dir,
        threshold=0.25,
        match_threshold=0.25,
        min_comsize=3,
        save=True,
    ):
        """
        Constructor

        :param network_files_list: A list of snapshot .edges files, sorted chronologically
        :param output_dir: Path to the directory to save results
        :param threshold: the tolerance required in order to merge communities (for Angel)
        :param match_threshold: the tolerance for matching communities across time
        :param min_comsize: minimum desired community size
        :param save: (True|False) whether to output the result on file or not
        """

        self.network_files = network_files_list
        self.output_dir = output_dir
        self.threshold = threshold
        self.match_threshold = match_threshold

        if self.threshold < 1:
            self.min_community_size = max(
                [3, min_comsize, int(1.0 / (1 - self.threshold))]
            )
        else:
            self.min_community_size = min_comsize

        self.save = save
        # Ensure output directory exists
        if self.save and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.matches_file = os.path.join(self.output_dir, "ct_matches.csv")
        self.slices_ids = []
        self.snapshot_to_coms = {}

    def __load_graph_from_edgelist(self, filepath):
        """
        Reads a single snapshot .edges file (space-separated) and returns
        an igraph Graph and the snapshot ID.

        :param filepath: path to the .edges file
        :return: (igraph.Graph, str_snapshot_id)
        """

        # Extract snapshot ID from filename, e.g., "mergesplit.t01.edges" -> "t01"
        basename = os.path.basename(filepath)
        try:
            # Assumes format like 'name.tXX.edges'
            snapshot_id = basename.split(".")[-2]  # e.g., 't01'
        except IndexError:
            # Fallback for simple names
            snapshot_id = os.path.splitext(basename)[0]

        edge_list = []
        vertices = set()

        with open(filepath) as f:
            for line in f:
                parts = line.rstrip().split()  # Use split() for space-separated
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    edge_list.append((u, v))
                    vertices.add(u)
                    vertices.add(v)

        sorted_vertices = sorted(list(vertices))
        g = igraph.Graph()
        g.add_vertices(sorted_vertices)
        g.add_edges(edge_list)

        return g, snapshot_id

    @timeit
    def execute(self):
        """
        Execute ArchAngel algorithm on a list of snapshot files
        """

        print("Phase 1: Detecting communities in each snapshot...")
        # Loop over the provided list of snapshot files
        for filepath in tqdm.tqdm(self.network_files, ncols=50):
            graph, snapshot = self.__load_graph_from_edgelist(filepath)
            self.slices_ids.append(snapshot)  # Keep track of snapshot order

            # Construct the output file path as requested
            input_basename = os.path.basename(filepath)
            output_basename = input_basename.replace(".edges", ".angel_communities.txt")
            output_filepath = os.path.join(self.output_dir, output_basename)

            # Initialize and run the ANGEL algorithm for this snapshot
            ag = an.Angel(
                None,
                threshold=self.threshold,
                min_comsize=self.min_community_size,
                save=self.save,
                outfile_name=output_filepath,  # Use the specific output path
                dyn=graph,
                verbose=False,
            )

            # Store the resulting communities
            self.snapshot_to_coms[snapshot] = ag.execute()

        print("\nPhase 2: Matching communities across time...")
        # --- Matching Step ---
        with open(self.matches_file, "w") as fout:
            fout.write("snapshot_from,snapshot_to,cid_from,cid_to\n")

            # Matching
            for t, fr in enumerate(self.slices_ids[:-1]):
                if t < len(self.slices_ids) - 1:
                    to = self.slices_ids[t + 1]  # Get the next snapshot
                    mts = self.__tpr_match(fr, to)

                    # Output cross-time matches
                    for past, future in iteritems(mts):
                        for c_future in future:
                            fout.write("%s,%s,%s,%s\n" % (fr, to, past, c_future))

        print("Matching complete.")
        return self.snapshot_to_coms

    def __tpr_match(self, fr, to):
        """
        Apply F1-merge to ego-network based micro communities

        :param community_to_nodes: dictionary <community_id, node_list>
        """

        community_to_nodes_from = self.snapshot_to_coms[fr]
        community_to_nodes_to = self.snapshot_to_coms[to]

        community_events = {}

        if len(community_to_nodes_from) == 0 or len(community_to_nodes_to) == 0:
            return community_events

        # From past to future
        node_to_com_to = {
            n: cid for cid, nlist in community_to_nodes_to.items() for n in nlist
        }

        # cycle over micro communities
        for c in community_to_nodes_from:
            actual_community = community_to_nodes_from[c]
            matches = [
                node_to_com_to[n] for n in actual_community if n in node_to_com_to
            ]
            most_common_coms = {
                cid: cid_count for cid, cid_count in Counter(matches).most_common()
            }

            if len(most_common_coms) > 0:
                max_ct = list(most_common_coms.values())[0]

                similarity = float(max_ct) / len(actual_community)
                if similarity >= self.match_threshold:
                    for cf in most_common_coms.keys():
                        if c not in community_events:
                            community_events[c] = [cf]
                        else:
                            community_events[c].append(cf)
                            community_events[c] = list(set(community_events[c]))

        # From future to past
        node_to_com_from = {
            n: cid for cid, nlist in community_to_nodes_from.items() for n in nlist
        }

        # cycle over micro communities
        for c in community_to_nodes_to:
            actual_community = community_to_nodes_to[c]
            matches = [
                node_to_com_from[n] for n in actual_community if n in node_to_com_from
            ]
            most_common_coms = {
                cid: cid_count for cid, cid_count in Counter(matches).most_common()
            }

            if len(most_common_coms) > 0:
                max_ct = list(most_common_coms.values())[0]

                similarity = float(max_ct) / len(actual_community)
                if similarity >= self.match_threshold:
                    for cf in most_common_coms:
                        if cf not in community_events:
                            community_events[cf] = [c]
                        else:
                            community_events[cf].append(c)
                            community_events[cf] = list(set(community_events[cf]))

        return community_events


# ===============================================
# Main execution block
# ===============================================
if __name__ == "__main__":

    # 1. Define Paths (using raw strings 'r' to handle backslashes)
    input_dir = r"C:\Users\91850\OneDrive\Desktop\waste\akarsh"
    output_dir = r"C:\Users\91850\OneDrive\Desktop\waste\akarsh\angel"

    # 2. Find and sort snapshot files
    # glob.glob might not return files in numerical order (e.g., t10 before t2)
    # We use sorted() to ensure they are processed in order: t01, t02, ..., t10
    file_pattern = os.path.join(input_dir, "mergesplit.t*.edges")
    snapshot_files = sorted(glob.glob(file_pattern))

    if not snapshot_files:
        print("Error: No files found matching pattern: %s" % file_pattern)
        print("Please check the 'input_dir' path and file names.")
        sys.exit(1)

    print("Found %d snapshot files to process:" % len(snapshot_files))
    for f in snapshot_files:
        print("  %s" % os.path.basename(f))

    # 3. Initialize and run ArchAngel
    # Using default parameters from the original class __init__
    arch = ArchAngel(
        network_files_list=snapshot_files,
        output_dir=output_dir,
        threshold=0.25,
        match_threshold=0.25,
        min_comsize=3,
        save=True,
    )

    print("\nStarting ArchAngel execution...")
    arch.execute()

    print("\n------------------------------------")
    print("Execution finished.")
    print("Community files saved to: %s" % output_dir)
    print("Community matches file saved to: %s" % arch.matches_file)
    print("------------------------------------")
