Density Peaks Advanced clustering
=================================

Status of the `scikit-learn`_ compatibility test:

.. image:: https://github.com/mariaderrico/DPA/actions/workflows/runpytest.yml/badge.svg?branch=master
   :alt: scikit-learn compatibility test status on GitHub Actions
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
    |-- DP/                              # Auxiliary package with the DP clustering implementation.
    |-- docs/                            # Documentation files.
    |-- Examples/                        # Auxiliary scripts for the examples generations.
    |-- DPA_analysis.ipynb               # Use-case example for DPA.
    |-- DPA_comparison-all.ipynb         # Performance comparison with other clustering methods.
    |-- README.rst
    |-- compile.sh
    |-- setup.py
    |-- src/                             # Source files for DPA, PAk and twoNN algorithms.


Source files
------------

The source Python codes are stored inside the ``src`` folder.

::

    .
    |-- ...
    |-- src/
    |   |-- Pipeline/
    |       |-- __init__.py
    |       |-- DPA.py           # Python module implementing the DPA
    |       |                    # clustering algorithm.
    |       |
    |       |-- _DPA.pyx         # Cython extension of the DPA module.
    |       |
    |       |-- PAk.py           # Python module implementing the PAk
    |       |                    # density estimator.
    |       |
    |       |-- _PAk.pyx         # Cython extension of the PAk module.
    |       |
    |       |-- twoNN.py         # Python module implementing the TWO-NN
    |                            # algorithm for the ID calculation.
    |
    |-- ...   

Documentation files
-------------------

Full documentation about the Python codes developed and the how-to instructions is created in the ``docs`` folder using `Sphinx`.
Complete documentation for DPA is available on the  `Read The Docs <https://dpaclustering.readthedocs.org>`_ website.

Jupyter notebooks
-----------------

Examples of how-to run the ``DPA``, ``PAk`` and ``twoNN`` modules are provided as Jupyter notebook in ``DPA_analysis.ipynb``. Additional useful use-cases are available in ``DPA_comparison-all.ipynb``, which include a performance comparison with the following clustering methods: Bayesian Gaussian Mixture, HDBSCAN, Spectral Clustering and Density Peaks.

Both jupyter notebooks are also available as Python script (saved using `jupytext`_) in the ``jupytext`` folder.
::

    .
    |-- ...
    |-- DPA_analysis.ipynb               # Use-case example for DPA.
    |-- DPA_comparison-all.ipynb         # Performance comparison with
    |                                    # other clustering methods.
    |    
    |-- ...
    |-- jupytext/
    |   |-- DPA_analysis.py              # DPA_analysis.ipynb saved as
    |   |                                # Python script.
    |   |-- DPA_comparison-all.py        # DPA_comparison-all.ipynb
    |                                    # saved as Python script.


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

Install required dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DPA package depends on ``easycython``, that can be installed using ``conda`` or ``pip``.
Note that it is possible to check which packages are installed with the ``pip freeze`` command.


Installing released code from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the latest version from the GitHub repository via::

    pip install git+https://github.com/mariaderrico/DPA

Installing development code from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following commands to download the DPA source code::

    git clone https://github.com/mariaderrico/DPA.git

Install DPA with the following commands::

    cd DPA
    . compile.sh



Citing
------

If you have used this codebase in a scientific publication and wish to cite the algorithm, please cite our paper in Information Sciences.


    `M. d'Errico, E. Facco, A. Laio, A. Rodriguez, Information Sciences, Volume 560, June 2021, 476-492`_


.. code:: bibtex

    @article{DERRICO2021476,
      title = {Automatic topography of high-dimensional data sets by non-parametric density peak clustering},
      journal = {Information Sciences},
      volume = {560},
      pages = {476-492},
      year = {2021},
      issn = {0020-0255},
      doi = {https://doi.org/10.1016/j.ins.2021.01.010},
      url = {https://www.sciencedirect.com/science/article/pii/S0020025521000116},
      author = {Maria d’Errico and Elena Facco and Alessandro Laio and Alex Rodriguez},
      }



.. References

.. _`scikit-learn`: https://scikit-learn.org/stable/
.. _`M. d'Errico, E. Facco, A. Laio, A. Rodriguez, Information Sciences, Volume 560, June 2021, 476-492`: https://www.sciencedirect.com/science/article/pii/S0020025521000116?dgcid=author
.. _`arXiv`: https://arxiv.org/abs/1802.10549v2
.. _`Computing the free energy without collective variables`: https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00916 
.. _`Estimating the intrinsic dimension of datasets by a minimal neighborhood information`: https://export.arxiv.org/pdf/1803.06992 
.. _`Clustering by fast search and find of density peaks`: http://science.sciencemag.org/content/344/6191/1492.full.pdf
.. _`github DPA repository`: https://github.com/mariaderrico/DPA.git
.. _`Anaconda`: https://www.anaconda.com/download/#windows
.. _`git`: https://git-scm.com
.. _`jupytext`: https://pypi.org/project/jupytext/
