# G/SG Morph

## The Graph/Subgraph Isomorphism Library

G/SG Morph is an implementation of [Calude et al.'s "QUBO formulations for the graph isomorphism problem and related problems"](https://doi.org/10.1016/j.tcs.2017.04.016) (Quantum Unconstrained Binary Optimization) QUBO's for finding graph, subgraph, and induced subgraph isomorphisms on quantum annealers.

## Dependencies

G/SG Morph requires NetworkX to be installed.

Please follow the instructions [here](https://pyqubo.readthedocs.io/en/latest/getting_started.html) to install  [here](https://networkx.org/documentation/stable/install.html) NetworkX.

## High Level Overview

G/SG Morph contains three core functions, `graph_isomorphism()`, `subgraph_isomorphism()`, and `translate_sample()`.

Both `graph_isomorphism()` and `subgraph_isomorphism()` take two [NetworkX](https://networkx.org/) graphs (a "graph to embed" onto a  "target graph") and will generate a QUBO suitable for running on a simulated annealer such as [D-Wave Neal](https://github.com/dwavesystems/dwave-neal) or actual hardware.

The above functions also return a dictionary that, in conjunction with a sample from an annelear can be translated into a dictionary that maps nodes from the graph to embed to the target graph with the help of `translate_sample()`.

`subgraph_isomorphism()` also has an additional `induced` argument that can be set to `True` indicating that you would to generate a QUBO for the Induced Subgraph Isomorphism problem.

## Examples

Please refer to the Jupyter Notebook `gsg_morph_demo.ipynb` for usage examples.