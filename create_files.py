from modules.graph import Graph


graph = Graph()

dataset_path = ''

graph.read_from_edgelist(f'{dataset_path}/edges_remove_3x_nodes_degree_1.csv')


# Create node list for link prediction task
# Graph.write_nodelist(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      )

# Create node list for AS_hegemony classification task
# Graph.write_nodelist(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      [
#                          'AS_hegemony'
#                      ])

# Create node list for AS_rank_continent classification task. The labels that contains few records are merged in the same label
# Graph.write_nodelist(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      [
#                          'AS_rank_continent_None', 'AS_rank_continent_Africa', 'AS_rank_continent_Asia',
#                          'AS_rank_continent_Europe', 'AS_rank_continent_North America', 'AS_rank_continent_Oceania',
#                          'AS_rank_continent_South America'
#                      ],
#                      True)

# Create node list for link prediction task using the node2vec embeddings
# Graph.write_nodelist(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node2vec-embeddings16-10-100.txt',
#                      )

# Create node list for link prediction task using the bgp2vec embeddings
# Graph.write_nodelist_with_subset(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/bgp2vec-embeddings.txt',
#                      )

# Create node list for AS_hegemony classification task using the node2vec embeddings
# Graph.write_nodelist(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      [
#                          'AS_hegemony'
#                      ],
#                      False,
#                      f'{dataset_path}/node2vec-embeddings16-10-100.txt')

# Create node list for AS_rank_continent classification task using the node2vec embeddings. The labels that contains few records are merged in the same label
# Graph.write_nodelist(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      [
#                          'AS_rank_continent_None', 'AS_rank_continent_Africa', 'AS_rank_continent_Asia',
#                          'AS_rank_continent_Europe', 'AS_rank_continent_North America', 'AS_rank_continent_Oceania',
#                          'AS_rank_continent_South America'
#                      ],
#                      True,
#                      f'{dataset_path}/node2vec-embeddings16-10-100.txt')

# Create node list for AS_hegemony classification task using the deepwalk embeddings
# Graph.write_nodelist(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      [
#                          'AS_hegemony'
#                      ],
#                      False,
#                      f'{dataset_path}/deepwalk-embeddings16-10-100.txt')

# Create node list for AS_rank_continent classification task using the deepwalk embeddings. The labels that contains few records are merged in the same label
# Graph.write_nodelist(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      [
#                          'AS_rank_continent_None', 'AS_rank_continent_Africa', 'AS_rank_continent_Asia',
#                          'AS_rank_continent_Europe', 'AS_rank_continent_North America', 'AS_rank_continent_Oceania',
#                          'AS_rank_continent_South America'
#                      ],
#                      True,
#                      f'{dataset_path}/deepwalk-embeddings16-10-100.txt')

# Create node list for AS_hegemony classification task using the bgp2vec embeddings
# Graph.write_nodelist_with_subset(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      [
#                          'AS_hegemony'
#                      ],
#                      f'{dataset_path}/bgp2vec-embeddings.txt')

# Create node list for AS_rank_continent classification task using the bgp2vec embeddings. The labels that contains few records are merged in the same label
# Graph.write_nodelist_with_subset(graph.get_graph(),
#                      dataset_path,
#                      f'{dataset_path}/node_features.csv',
#                      [
#                          'AS_rank_continent_None', 'AS_rank_continent_Africa', 'AS_rank_continent_Asia',
#                          'AS_rank_continent_Europe', 'AS_rank_continent_North America', 'AS_rank_continent_Oceania',
#                          'AS_rank_continent_South America'
#                      ],
#                      True,
#                      f'{dataset_path}/bgp2vec-embeddings.txt')
