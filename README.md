# Privacy-Preserving Community Detection in Dynamic Networks

A comprehensive framework for generating synthetic dynamic networks with ground truth community structure, applying community detection algorithms, and protecting network privacy by hiding sensitive overlapping nodes.

## 📋 Overview

This project implements a complete pipeline for analyzing and protecting privacy in dynamic networks:

1. **Data Generation**: Generate synthetic dynamic networks with embedded communities using C++ generators
2. **Community Detection**: Apply two state-of-the-art algorithms:
   - **ArchAngel**: Overlapping community detection for dynamic networks
   - **TILES**: Online iterative community discovery
3. **Privacy Protection**: Hide overlapping nodes to preserve community privacy while maintaining network structure

## 📁 Project Structure

```
privacy_preserving_in_dynamic_networks/
├── README.md                          # This file
├── DataGeneration.py                  # Step 1: Generate synthetic dynamic networks
├── RunTiles.py                        # Step 2a: Execute TILES algorithm
├── Analysis.py                        # Step 3: Compare algorithm results with ground truth
├── dynamic_network_with_timestamps.csv # Generated network data
├── communities.csv                    # Ground truth communities
├── ground_truth_dynamic.gexf          # Network in GEXF format
│
├── akarsh/                            # ArchAngel implementation & results
│   ├── angel/                         # ArchAngel algorithm source code
│   ├── src/                           # C++ source files for network generation
│   ├── mergesplit.t*.comm             # Detected communities (10 timesteps)
│   ├── mergesplit.t*.edges            # Network snapshots
│   ├── angelArchAngel_coms_t*.txt     # ArchAngel results by timestep
│   └── evaluate_communities.py        # Evaluation metrics
│
├── TILES/                             # TILES algorithm implementation
│   ├── tiles/                         # Core TILES algorithm
│   ├── tiles_input.tsv                # Input data for TILES
│   └── tiles_output/                  # TILES detection results
│
└── hiding/                            # Privacy protection module
    ├── Bihdyn.py                      # BIH Dynamic - overlapping node hiding
    ├── angel/                         # ArchAngel for privacy-aware detection
    ├── BIH_Results/                   # Modified networks with hidden nodes
    │   ├── modified_networks/         # Network snapshots after hiding
    │   └── reports/                   # Privacy modification reports
    └── GroundTruth/                   # Copy of original ground truth
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- NetworkX
- Pandas
- NumPy
- scikit-learn
- C++ compiler (for data generation if building from source)

### Installation

```bash
# Clone the repository
git clone https://github.com/Akarsh2005/privacy_preserving_in_dynamic_networks.git
cd privacy_preserving_in_dynamic_networks

# Install dependencies
pip install -r requirements.txt
```

## 📊 Workflow

### Step 1: Generate Synthetic Dynamic Networks

**File**: `DataGeneration.py`

Generates LFR-style networks with temporal evolution and embedded communities:

```bash
python DataGeneration.py
```

**Output**:

- `dynamic_network_with_timestamps.csv` - Edge list with timestamps
- `communities.csv` - Ground truth community assignments
- `ground_truth_dynamic.gexf` - Network visualization format

**Parameters** (configurable in script):

- Number of nodes and timesteps
- Average degree
- Mixing parameter (community cohesion)
- Overlap percentage
- Community size range

### Step 2a: Community Detection with ArchAngel

**Directory**: `akarsh/`

ArchAngel detects overlapping communities in dynamic networks:

```bash
cd akarsh
python evaluate_communities.py
```

**Output**:

- `angelArchAngel_coms_t*.txt` - Detected communities per timestep
- `mergesplit.t*.comm` - Ground truth communities per timestep
- Evaluation metrics and statistics

### Step 2b: Community Detection with TILES

**File**: `RunTiles.py`

TILES performs online iterative community discovery:

```bash
python RunTiles.py
```

**Output**:

- `tiles_output/` - Community detection results
- Comparison with ground truth metrics

### Step 3: Analysis & Comparison

**File**: `Analysis.py`

Comprehensive analysis comparing both algorithms against ground truth:

```bash
python Analysis.py
```

**Metrics computed**:

- Normalized Mutual Information (NMI)
- Precision and Recall
- Community sizes and distributions
- Algorithm performance over time

## 🔒 Privacy Protection Module

### Step 4: Hide Overlapping Nodes

**Directory**: `hiding/`

The privacy preservation pipeline identifies and hides sensitive overlapping nodes:

```bash
cd hiding
python Bihdyn.py  # Based Importance Hiding for Dynamic Networks
```

**Process**:

1. Load detected communities from ArchAngel
2. Identify overlapping nodes (nodes in multiple communities)
3. Modify network edges to obscure community structure
4. Save anonymized network snapshots
5. Re-run community detection on modified networks

**Output**:

- `BIH_Results/modified_networks/` - Anonymized network snapshots
- `BIH_Results/reports/` - Privacy modification details
- Analysis of privacy-utility tradeoffs

### Privacy Metrics

The hiding process protects:

- **Node Privacy**: Overlapping nodes are obscured
- **Community Structure**: Original communities become harder to detect
- **Utility**: Network functionality is preserved

## 📈 Data Format Specifications

### Dynamic Network Input

CSV format with columns:

```
source,target,timestamp
```

### Community Files (ArchAngel format)

Each line represents one community:

```
node1 node2 node3 ...
```

### TILES Input Format (TSV)

Tab-separated without header:

```
node_id0	node_id1	timestamp
```

## 📊 Key Files Description

| File                             | Purpose                                |
| -------------------------------- | -------------------------------------- |
| `DataGeneration.py`              | LFR-style synthetic network generator  |
| `RunTiles.py`                    | TILES algorithm executor and analyzer  |
| `Analysis.py`                    | Cross-algorithm comparison and metrics |
| `akarsh/evaluate_communities.py` | ArchAngel evaluation metrics           |
| `hiding/Bihdyn.py`               | Privacy-preserving edge modification   |
| `communities.csv`                | Ground truth for validation            |

## 🧪 Algorithms

### ArchAngel (Dynamic Overlapping Communities)

- Detects overlapping communities in dynamic networks
- Supports community birth, death, and evolution
- Output: Community membership per timestep
- Source: `akarsh/angel/`

### TILES (Tracking In Labeled Edge Streams)

- Online iterative community discovery
- Handles edge additions and removals
- Tracks community evolution over time
- Source: `TILES/tiles/`

### BIH (Based Importance Hiding)

- Privacy-aware edge modification strategy
- Targets overlapping nodes (most sensitive)
- Preserves network properties while hiding structure
- Source: `hiding/Bihdyn.py`

## 📝 Output Metrics

### Community Detection Quality

- **NMI** (Normalized Mutual Information): 0-1 scale
- **Precision**: Correctness of detected communities
- **Recall**: Completeness of detection
- **F-Score**: Harmonic mean of precision/recall

### Privacy Metrics

- **Nodes Hidden**: Number of overlapping nodes obscured
- **Edges Modified**: Network changes for privacy
- **Detectability Change**: Algorithm performance degradation
- **Utility Preservation**: Original network properties retained

## 🔧 Configuration

Key parameters in `DataGeneration.py`:

```python
NUM_NODES = 1000                  # Network size
NUM_TIMESTEPS = 10               # Temporal snapshots
AVG_DEGREE = 10                  # Average node degree
MIXING_PARAMETER = 0.3           # Community mixing
OVERLAP_PERCENTAGE = 0.1         # % of overlapping nodes
```

## 📚 References

### Papers & Algorithms

- **ArchAngel**: Community detection in dynamic networks with overlaps
- **TILES**: "Tiles: an online algorithm for community discovery in dynamic social networks" (Rossetti et al., 2016)
- **LFR Networks**: Lancichinetti, Fortunato, and Radicchi benchmark

### Citation

If you use this framework, please cite:

```bibtex
@inproceedings{privacy_dynamic_2025,
  title={Privacy-Preserving Community Detection in Dynamic Networks},
  author={Akarsh},
  year={2025}
}
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional privacy mechanisms (differential privacy, k-anonymity)
- More community detection algorithms
- Visualization tools
- Performance optimizations
- Extended documentation

## 📄 License

See LICENSE file for details.

## 👤 Author

**Akarsh**

- Repository: [privacy_preserving_in_dynamic_networks](https://github.com/Akarsh2005/privacy_preserving_in_dynamic_networks)

## 🆘 Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure all Python dependencies are installed:

```bash
pip install networkx pandas numpy scikit-learn
```

### Issue: Data generation takes long

**Solution**: Reduce `NUM_NODES` or `NUM_TIMESTEPS` in DataGeneration.py

### Issue: TILES algorithm crashes

**Solution**: Ensure timestamps are sorted in ascending order

## 📞 Support

For issues or questions:

1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error messages and reproduction steps

---

**Last Updated**: December 2025
**Status**: Active Development
