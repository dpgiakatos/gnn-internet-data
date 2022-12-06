# Dataset
## Create files for benchmarking

To create the nodes and edge CSV files for the benchmarking tests use the `create_files.py` file. In this file you should set the `dataset_path`.
The nodes CSV file can be created using the function:
- `Graph.write_nodelist`: Create nodes CSV files for Internet features, Node2Vec, and DeepWalk embeddings.
- `Graph.write_nodelist_with_subset`: Create nodes CSV files only for BGP2Vec embeddings.

Both function takes the same parameters and creates two files, one with the features and one without features. The parameters are:
- `graph`: The graph instance.
- `path`: The directory of the dataset that we want to export the nodes CSV files.
- `features_filename`: The path for the Internet features.
- `label`: Default `None`. The labels in a list of strings for the node classification
- `to_merge`: Default `False`. If `True` the labels that contains few records are merged in the same label.
- `embeddings_filename`: Default `None`. The path for the file that contains the node embeddings.

The `create_files.py` file contains some examples.

## Directory structure

The structure for the benchmarking should be in the format that presented bellow.
The `edge.csv` file is the `edges_remove_3x_nodes_degree_1.csv` and the `node.csv` is the file that created using the functions that presented above.
The `meta.yaml` file is the file that exist in the current directory.

```bash
├── dataset                                
    ├── link-prediction
    │   ├── baseline
    │   │   └── node-features
    │   │       ├── edges.csv
    │   │       ├── meta.yaml
    │   │       └── nodes.csv
    │   ├── bgp2vec
    │   │   └── embeddings
    │   │       ├── edges.csv
    │   │       ├── meta.yaml
    │   │       └── nodes.csv
    │   ├── gnn
    │   │   ├── with-features
    │   │   │   ├── edges.csv
    │   │   │   ├── meta.yaml
    │   │   │   └── nodes.csv
    │   │   └── without-features
    │   │       ├── edges.csv
    │   │       ├── meta.yaml
    │   │       └── nodes.csv
    │   └── node2vec
    │       ├── embeddings16-10-100
    │       │   ├── edges.csv
    │       │   ├── meta.yaml
    │       │   └── nodes.csv
    │       ├── embeddings16-50-200
    │       │   ├── edges.csv
    │       │   ├── meta.yaml
    │       │   └── nodes.csv
    │       ├── embeddings16-wl40-nm150
    │       │   ├── edges.csv
    │       │   ├── meta.yaml
    │       │   └── nodes.csv
    │       └── embeddings16-wl4-nm20
    │           ├── edges.csv
    │           ├── meta.yaml
    │           └── nodes.csv
    └── node-classification
        ├── AS_hegemony
        │   ├── baseline
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── bgp2vec-embeddings
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-10-100
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-50-200
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl40-nm150
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl4-nm20
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── with-features
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   └── without-features
        │       ├── edges.csv
        │       ├── meta.yaml
        │       └── nodes.csv
        ├── AS_rank_continent
        │   ├── baseline
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── bgp2vec-embeddings
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-10-100
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-50-200
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl40-nm150
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl4-nm20
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── with-features
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   └── without-features
        │       ├── edges.csv
        │       ├── meta.yaml
        │       └── nodes.csv
        ├── peeringDB_info_ratio
        │   ├── baseline
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── bgp2vec-embeddings
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-10-100
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-50-200
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl40-nm150
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl4-nm20
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── with-features
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   └── without-features
        │       ├── edges.csv
        │       ├── meta.yaml
        │       └── nodes.csv
        ├── peeringDB_info_scope
        │   ├── baseline
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── bgp2vec-embeddings
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-10-100
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-50-200
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl40-nm150
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl4-nm20
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── with-features
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   └── without-features
        │       ├── edges.csv
        │       ├── meta.yaml
        │       └── nodes.csv
        ├── peeringDB_info_type
        │   ├── baseline
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── bgp2vec-embeddings
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-10-100
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-50-200
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl40-nm150
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── node2vec-embeddings16-wl4-nm20
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   ├── with-features
        │   │   ├── edges.csv
        │   │   ├── meta.yaml
        │   │   └── nodes.csv
        │   └── without-features
        │       ├── edges.csv
        │       ├── meta.yaml
        │       └── nodes.csv
        └── peeringDB_policy_general
            ├── baseline
            │   ├── edges.csv
            │   ├── meta.yaml
            │   └── nodes.csv
            ├── bgp2vec-embeddings
            │   ├── edges.csv
            │   ├── meta.yaml
            │   └── nodes.csv
            ├── node2vec-embeddings16-10-100
            │   ├── edges.csv
            │   ├── meta.yaml
            │   └── nodes.csv
            ├── node2vec-embeddings16-50-200
            │   ├── edges.csv
            │   ├── meta.yaml
            │   └── nodes.csv
            ├── node2vec-embeddings16-wl40-nm150
            │   ├── edges.csv
            │   ├── meta.yaml
            │   └── nodes.csv
            ├── node2vec-embeddings16-wl4-nm20
            │   ├── edges.csv
            │   ├── meta.yaml
            │   └── nodes.csv
            ├── with-features
            │   ├── edges.csv
            │   ├── meta.yaml
            │   └── nodes.csv
            └── without-features
                ├── edges.csv
                ├── meta.yaml
                └── nodes.csv
```

## Results: Detailed distributions

Here the detailed distributions per feature are presented, in Histogram format for the categorical ones and CDF for the numerical ones.
The following figures depict the distributions of values for all ASes and for RIPE NCC monitors, along different dimensions. 

You can click on a figure to zoom in. All images can be found in the `./figures/` folder.  

**Location related dimensions**

&nbsp;|RIR region|Location (continent)|&nbsp;| &nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp; |![](./figures/EDA_Histogram_AS_rank_source.png?raw=true)| ![](./figures/EDA_Histogram_AS_rank_source.png?raw=true)|&nbsp;|&nbsp;


**Network size dimensions**

Customer cone (#ASNs) | Customer cone (#prefixes) | Customer cone (#addresses) | AS hegemony | &nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/EDA_CDF_AS_rank_numberAsns.png?raw=true)|![](./figures/EDA_CDF_AS_rank_numberPrefixes.png?raw=true)|![](./figures/EDA_CDF_AS_rank_numberAddresses.png?raw=true)|![](./figures/EDA_CDF_AS_hegemony.png?raw=true)|&nbsp;


**Topology related dimensions**

#neighbors (total)|#neighbors (peers)|#neighbors (customers)|#neighbors (providers)|&nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/EDA_CDF_AS_rank_total.png?raw=true)|![](./figures/EDA_CDF_AS_rank_peer.png?raw=true)|![](./figures/EDA_CDF_AS_rank_customer.png?raw=true)|![](./figures/EDA_CDF_AS_rank_provider.png?raw=true)|&nbsp;



**IXP related dimensions**

&nbsp;|#IXPs (PeeringDB)|#facilities (PeeringDB)|Peering policy (PeeringDB)|&nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp;|![](./figures/EDA_CDF_peeringDB_ix_count.png?raw=true)|![](./figures/EDA_CDF_peeringDB_fac_count.png?raw=true)|![](./figures/EDA_Histogram_peeringDB_policy_general.png?raw=true)|&nbsp;


**Network type dimensions**

Network type (PeeringDB)|Traffic ratio (PeeringDB)|Traffic volume (PeeringDB)|Scope (PeeringDB)|Personal ASN
:---:|:---:|:---:|:---:|:---:
![](./figures/EDA_Histogram_peeringDB_info_type.png?raw=true)|![](./figures/EDA_Histogram_peeringDB_info_ratio.png?raw=true)|![](./figures/EDA_Histogram_peeringDB_info_traffic.png?raw=true)|![](./figures/EDA_Histogram_peeringDB_info_scope.png?raw=true)|![](./figures/EDA_Histogram_is_personal_AS.png?raw=true)
