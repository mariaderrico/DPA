#!/bin/bash

# create python scripts out of the jupyter notebooks
# 
# How-to-run:
#
# cd DPA
# ./run_ipynb2py_versioning.sh

for file in *.ipynb; do 
  echo "$file";
  NAME=`echo "${file##*/}" | cut -d'.' -f1`;
  echo $NAME;
  echo "jupytext --output "${NAME}".py" ${file};
  eval $(echo -e "jupytext --output "${NAME}".py" ${file});
  mv ${NAME}.py jupytext/;
  done;
