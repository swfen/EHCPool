# EHCPool
Edge-aware Hard Clustering Graph Pooling for dominant edge feature

This is a first graph pooling operator that targeted at dominant edge feature(compare to the node),and also has clustering capabilities.

In order to craft a graph clustering pooling kernel adapted to dominant edge features, we innovate thoroughly the three key steps (from start to finish) in the graph pooling process. 

In graph neural networks with edge features, this pool kernel is the generic module, so we release the modular source code directly, rather than the full code of the paper.

The project follows the Pytorch library at https://pytorch-geometric.readthedocs.io/en/latest/
If you want to run the project directly, it is recommended that you follow the settings of this library
