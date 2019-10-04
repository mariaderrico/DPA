#!/bin/bash

cd DPA
echo=$(f2py -c NRmaxL.f90 -m NR)
echo=$(easycython _DPA.pyx)
echo=$(easycython _PAk.pyx)
cd ..
echo=$(python setup.py install) 
