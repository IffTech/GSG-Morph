# G/SG Morph

## The Graph/Subgraph Isomorphism Library for Quantum Annealers

G/SG Morph is an implementation of [Calude, Dineen, and Hua's "QUBO formulations for the graph isomorphism problem and related problems"](https://doi.org/10.1016/j.tcs.2017.04.016) (Quantum Unconstrained Binary Optimization) QUBO's for finding graph, subgraph, and induced subgraph isomorphisms on quantum annealers.

G/SG Morph also contains, with the permission of Richard Hua, a copy of his implementation of the Graph Isomorphism QUBO from his thesis ["Adiabatic Quantum Computing with QUBO Formulations", Appendix E](https://researchspace.auckland.ac.nz/bitstream/handle/2292/31576/whole.pdf?sequence=2&isAllowed=y) which is used for benchmarking (see "Benchmarking" in this README).

## Installation

Clone this repository and run the following in the folder (and your choice of python environment!):

```
pip install .
```

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

## Benchmarking

Some benchmarking was conducted against Richard Hua's graph isomorphism QUBO generator and G/SG Morph's implementation using Erdos-Renyi graphs in Google Colab. The results and techniques can be found in the `Benchmarking` folder.