# G/SG Morph

## The Graph/Subgraph Isomorphism Library for Quantum Annealers

G/SG Morph is an implementation of [Calude, Dinneen, and Hua's "QUBO formulations for the graph isomorphism problem and related problems"](https://doi.org/10.1016/j.tcs.2017.04.016) (Quantum Unconstrained Binary Optimization) QUBO's for finding graph, subgraph, and induced subgraph isomorphisms on quantum annealers.

G/SG Morph also contains, with the permission of Richard Hua, a copy of his implementation of the Graph Isomorphism QUBO from his thesis ["Adiabatic Quantum Computing with QUBO Formulations", Appendix E](https://researchspace.auckland.ac.nz/bitstream/handle/2292/31576/whole.pdf?sequence=2&isAllowed=y) which is used for benchmarking (see "Benchmarking" in this README).

## Installation

You can either 

```
pip install gsgmorph
```

or clone this repository and run the following in the folder (and your choice of python environment!):

```
pip install .
```

## High Level Overview

G/SG Morph consists of two modules, `matrix_form` and `pyqubo_form`, both of which contain three core functions, `graph_isomorphism()`, `subgraph_isomorphism()`, and `translate_sample()` that accomplish identical tasks but are implemented differently. 

`matrix_form` relies on the usage of dictionaries to provide a matrix representing the QUBO, while `pyqubo_form` uses the [PyQUBO](https://github.com/recruit-communications/pyqubo) library to express QUBOs. Note that `pyqubo_form` has been intentionally programmed to follow the math presented in Calude, Dineen, and Hua's paper as closely as possible (with minor adjustments made to satisfy Python syntax). 

Both `graph_isomorphism()` and `subgraph_isomorphism()` take two [NetworkX](https://networkx.org/) graphs (a "graph to embed" onto a  "target graph") and will generate a QUBO suitable for running on a simulated annealer such as [D-Wave Neal](https://github.com/dwavesystems/dwave-neal) or actual hardware.

The above functions also return a dictionary that, in conjunction with a sample from an annelear, can be translated into a dictionary that maps nodes from the graph to embed to the target graph with the help of `translate_sample()`.

`subgraph_isomorphism()` also has an additional `induced` argument that can be set to `True` indicating that you would like to generate a QUBO for the Induced Subgraph Isomorphism problem.

## Examples

Please refer to the Jupyter Notebooks in the `examples` folder.

* `gsgmorph_matrix_form_demo.ipynb` shows the usage of the `matrix_form` module
* `gsgmorph_pyqubo_form_demo.ipynb` shows the usage of the `pyqubo_form` module

## Benchmarking

Some benchmarking was conducted against Richard Hua's graph isomorphism QUBO generator and G/SG Morph's `matrix_form` implementation using Erdos-Renyi graphs in Google Colab. The results and techniques can be found in the `Benchmarking` folder.

## Contributing

If you find a bug or have an idea to improve the library, please feel free to either make an Issue or a Pull Request with your suggested changes! If you are contributing code, please do note that this library attempts to follow the [PEP-8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/#package-and-module-names) as well as using [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

## Credits

Although all the QUBO formulations used come from [Calude, Dinneen, and Hua's "QUBO formulations for the graph isomorphism problem and related problems"](https://doi.org/10.1016/j.tcs.2017.04.016), this library would not have been possible without the following helpful sources:

* [Wangfan Fu's answer to 'What is the square of summation?'](https://math.stackexchange.com/questions/329344/what-is-the-square-of-summation) on the Math Stackexchange site
* [Dury and Matteo's code from their paper "A QUBO formulation for qubit allocation"](https://github.com/bdury/QUBO-for-Qubit-Allocation) [https://arxiv.org/pdf/2009.00140.pdf](https://arxiv.org/pdf/2009.00140.pdf) served as inspiration for the usage of the Python `product()` function. 
* [PyQUBO Documentation: Integration with D-Wave Ocean](https://pyqubo.readthedocs.io/en/latest/#integration-with-d-wave-ocean) for showing how to use [D-Wave Neal](https://docs.ocean.dwavesys.com/en/stable/docs_neal/sdk_index.html) with PyQUBO
* [SilentGhost's answer to "Reverse / invert a dictionary mapping"](https://stackoverflow.com/a/483833)
* [ars' answer to "Get difference between two lists"](https://stackoverflow.com/a/3462160)
* [Mccreesh, Prosser, Solnon, and Trimble's paper "When Subgraph Isomorphism is Really Hard, and Why This Matters for Graph Databases](https://hal.archives-ouvertes.fr/hal-01741928/document) for providing a graph to test the Induced Subgraph Isomorphism problem on
* [Dan D.'s answer to "what is diffrence between number and repeat in python timeit?"](https://stackoverflow.com/a/56763499)