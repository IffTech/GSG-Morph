# G/SG Morph

## The Graph/Subgraph Isomorphism Library for Quantum Annealers

G/SG Morph is an implementation of [Calude, Dineen, and Hua's "QUBO formulations for the graph isomorphism problem and related problems"](https://doi.org/10.1016/j.tcs.2017.04.016) (Quantum Unconstrained Binary Optimization) QUBO's for finding graph, subgraph, and induced subgraph isomorphisms on quantum annealers.

## High Level Overview

G/SG Morph consists of two modules, `matrix_form` and `pyqubo_form`, both of which contain three core functions, `graph_isomorphism()`, `subgraph_isomorphism()`, and `translate_sample()` that accomplish identical tasks but are implemented differently. 

`matrix_form` relies on the usage of dictionaries to provide a matrix representing the QUBO, while `pyqubo_form` uses the [PyQUBO](https://github.com/recruit-communications/pyqubo) library to express QUBOs.

Both `graph_isomorphism()` and `subgraph_isomorphism()` take two [NetworkX](https://networkx.org/) graphs (a "graph to embed" onto a  "target graph") and will generate a QUBO suitable for running on a simulated annealer such as [D-Wave Neal](https://github.com/dwavesystems/dwave-neal) or actual hardware.

The above functions also return a dictionary that, in conjunction with a sample from an annelear, can be translated into a dictionary that maps nodes from the graph to embed to the target graph with the help of `translate_sample()`.

`subgraph_isomorphism()` also has an additional `induced` argument that can be set to `True` indicating that you would like to generate a QUBO for the Induced Subgraph Isomorphism problem.

## Examples

Please refer to the Jupyter Notebooks in the `examples` folder.

* `gsgmorph_matrix_form_demo.ipynb` shows the usage of the `matrix_form` module
* `gsgmorph_pyqubo_form_demo.ipynb` shows the usage of the `pyqubo_form` module