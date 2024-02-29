# EHCPool
Edge-aware Hard Clustering Graph Pooling for dominant edge feature

This is a first graph pooling operator that targeted at dominant edge feature(compare to the node),and also has clustering capabilities.

In order to craft a graph clustering pooling kernel adapted to dominant edge features, we innovate thoroughly the three key steps (from start to finish) in the graph pooling process. For your better use and learning, please read the paper in

Please do not worry about the 'for'  loops that appear in the algorithm. These loops are not related to the sample size, but to the number of batch, the number of subgraphs and core nodes that are retained. The purpose of pooling is to obtain a limited number of subgraphs, so the computational complexity will not surge due to the use of large samples.

In graph neural networks with edge features, this pool kernel is the generic module, so we release the modular source code directly, rather than the full code of the paper.

The project follows the Pytorch library at https://pytorch-geometric.readthedocs.io/en/latest/
If you want to run the project directly, it is recommended that you follow the settings of this library
