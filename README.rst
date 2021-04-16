Density Peaks Advanced clustering
=================================

Status of the `scikit-learn`_ compatibility test:

.. image:: https://github.com/mariaderrico/DPA/actions/workflows/runpytest.yml/badge.svg?branch=master
   :alt: scikit-learn compatibility test status on Github Actions
   :target: https://github.com/mariaderrico/DPA/actions/workflows/runpytest.yml



The DPA package implements the Density Peaks Advanced (DPA) clustering algorithm as introduced in the paper "Automatic topography of high-dimensional data sets by non-parametric Density Peak clustering", published on `M. d'Errico, E. Facco, A. Laio, A. Rodriguez, Information Sciences, Volume 560, June 2021, 476-492`_  (also available on `arXiv`_).

The package offers the following features:

* Intrinsic dimensionality estimation by means of the `TWO-NN` algorithm, published in the `Estimating the intrinsic dimension of datasets by a minimal neighborhood information`_ paper.
* Adaptive k-NN Density estimation by means of the `PAk` algorithm, published in the `Computing the free energy without collective variables`_ paper.
* Advanced version of the `DP` clustering algorithm, published in the `Clustering by fast search and find of density peaks`_ paper, which includes an automatic search of cluster centers and assessment of statistical significance of the clusters  

.. contents::

Top-level directory layout
------------------------------

::

    cd DPA
    ls -l

::

    .
    |-- DP/                              # Python implementation of the Density Peaks clustering.
    |-- docs/                            # Documentation files. 
    |-- Examples/                        # Auxiliary scripts for the examples generations.
    |-- Pipeline/                        # Source files for DPA, PAk and twoNN algorithms.
    |-- DPA_analysis.ipynb               # Use-case example for DPA.
    |-- DPA_comparison-all.ipynb         # Performance comparison with other clustering methods.
    |-- README.rst
    |-- requirements.txt
    |-- config.sh
    |-- setup.py



Source files
------------

The source Python codes are stored inside the ``Pipeline`` folder::

    cd Pipeline
    ls -l

::

    .
    |-- ...
    |-- Pipeline/
    |   |-- __init__.py
    |   |-- DPA.py           # Python module implementing the DPA 
    |   |                    # clustering algorithm.
    |   |
    |   |-- _DPA.pyx         # Cython extension of the DPA module.
    |   |
    |   |-- PAk.py           # Python module implementing the PAk 
    |   |                    # density estimator.
    |   |
    |   |-- _PAk.pyx         # Cython extension of the PAk module.
    |   |                    
    |   |-- twoNN.py         # Python module implementing the TWO-NN
    |                        # algorithm for the ID calculation.                     
    |
    |-- ...   

Documentation files
-------------------

Full documentation about the Python codes developed and the how-to instructions is created in the ``docs`` folder using `Sphinx`.
The documentation in HTML format can be found in ``docs/_build/html/index.html``.
The ``DPApipeline.pdf`` is in the ``docs/_build/rinioh`` folder.


Jupyter notebooks
-----------------

Examples of how-to run the ``DPA``, ```PAk`` and ```twoNN`` modules are provided as Jupyter notebook in ``DPA_analysis.ipynb``. Additional useful use-cases are available in ``DPA_comparison-all.ipynb``, including a performance comparison with the following clustering methods: Bayesian Gaussian Mixture, HDBSCAN, Spectral Clustering and Density Peaks.

::

    .
    |-- ...
    |-- DPA_analysis.ipynb               # Use-case example for DPA.
    |-- DPA_comparison-all.ipynb         # Performance comparison with other clustering methods.
    |    
    |-- ...                                        
     

Getting started
---------------

The source code of DPA is on `github DPA repository`_. 

You need the ``git`` command in order to be able to clone it, and we
suggest you to use Python virtual environment in order to create a
controlled environment in which you can install DPA as
normal user avoiding conflicts with system files or Python libraries.

The following section documents the steps required to install DPA on a Linux or Windows/Mac computer.


Debian/Ubuntu
^^^^^^^^^^^^^

Run the following commands to create and activate a Python virtual environment with *python virtualenv*::

    apt-get install git python-dev virtualenv*
    virtualenv -p python3 venvdpa
    . venvdpa/bin/activate


Windows
^^^^^^^


A possible setup makes use of `Anaconda`_.
It has preinstalled and configured packages for data analysis and it is available on all major platforms. It uses *conda* as package manager, in addition to the standard pip.

A versioning control can be installed by downloading `git`_.

Run the following commands to activate the conda virtual environment::

    conda create -n venvdpa
    conda activate venvdpa

to list the available environments you can type ``conda info --envs``, and to deactivate an active environment use ``source deactivate``.


Installation
------------

Assuming you already have the Python virtual enviroment installed and activated on your machine, 
run the following commands to download the DPApipeline source code::

    git clone https://github.com/mariaderrico/DPA.git

Install DPA with the following commands::

    cd DPA
    . compile.sh 


Note that it is possible to check which packages are installed with the ``pip freeze`` command.


Quickstart
^^^^^^^^^^

A use-case example is provided in the DPA_analysis.ipynb jupyter notebook.


.. References

.. _`scikit-learn`: https://scikit-learn.org/stable/
.. _`M. d'Errico, E. Facco, A. Laio, A. Rodriguez, Information Sciences, Volume 560, June 2021, 476-492`: https://www.sciencedirect.com/science/article/pii/S0020025521000116?dgcid=author
.. _`arXiv`: https://arxiv.org/abs/1802.10549v2
.. _`Computing the free energy without collective variables`: https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00916 
.. _`Estimating the intrinsic dimension of datasets by a minimal neighborhood information`: https://export.arxiv.org/pdf/1803.06992 
.. _`Clustering by fast search and find of density peaks`: http://science.sciencemag.org/content/344/6191/1492.full.pdf
.. _`github DPApipeline repository`: https://github.com/mariaderrico/DPA.git
.. _`Anaconda`: https://www.anaconda.com/download/#windows
.. _`git`: https://git-scm.com
