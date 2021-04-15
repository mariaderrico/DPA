#!/bin/bash
echo=$(pip install numpy)
echo=$(pip install easycython)
echo=$(python setup.py install) 
cd Pipeline
echo=$(f2py -c NRmaxL.f90 -m NR)
echo=$(easycython _DPA.pyx)
echo=$(easycython _PAk.pyx)
cd ..
