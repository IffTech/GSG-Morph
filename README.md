# G/SG Morph

## The Graph/Subgraph Isomorphism Library

G/SG Morph is an implementation of [Calude et al.'s "QUBO formulations for the graph isomorphism problem and related problems"](https://doi.org/10.1016/j.tcs.2017.04.016) (Quantum Unconstrained Binary Optimization) QUBO's for finding graph and subgraph isomorphisms on quantum annealers.

The code in the `gsg_morph.py` file has variable names as well as a structure that closely follow the formulations in the original paper (with optimizations added where possible), made possible with [PyQUBO](https://github.com/recruit-communications/pyqubo). As a result, it should be easy to follow the math in the paper and the operation of the code. 

## Dependencies

G/SG Morph requires PyQUBO and NetworkX to be installed.

Please follow the instructions [here](https://pyqubo.readthedocs.io/en/latest/getting_started.html) to install PyQUBO and [here](https://networkx.org/documentation/stable/install.html) for NetworkX.

## High Level Overview

G/SG Morph contains three core functions, `graph_isomorphism()`, `subgraph_isomorphism()`, and `translate_sample()`.

Both `graph_isomorphism()` and `subgraph_isomorphism()` take two [NetworkX](https://networkx.org/) graphs (a "graph to embed" onto a  "target graph") and will generate a PyQUBO expression that can then be converted to QUBO, Ising, or BQM format suitable for running on a simulated annealer such as [D-Wave Neal](https://github.com/dwavesystems/dwave-neal) or actual hardware.

The above functions also return a dictionary that, in conjunction with a sample from an annelear translated by PyQUBO, can be translated into a dictionary that maps nodes from the graph to embed to the target graph with the help of `translate_sample()`.

`subgraph_isomorphism()` also has an additional `induced` argument that can be set to `True` indicating that you would to generate an expression for the Induced Subgraph Isomorphism problem.

## Examples

Please refer to the Jupyter Notebook `gsg_morph_demo.ipynb` for usage examples.