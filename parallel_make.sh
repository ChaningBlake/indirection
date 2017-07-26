#!/bin/bash

## To use (all cores): make -j
## To use (set cores): make -j 8
rm makefile

## Set variable command names
NUM_TESTS=50

## Create makefile header
echo -n -e "all: pre " >> makefile
for((i=1;i<=$NUM_TESTS;i++)) do
  echo -n -e "test$i " >> makefile
done
echo -n -e "\n" >> makefile

## Do any cleanup here
echo -n -e "pre:\n\t" >> makefile
echo -n -e "rm results.txt\n" >> makefile

## Create makefile body
for((i=1;i<=$NUM_TESTS;i++)) do
  echo -n -e "test$i:\n\t" >> makefile
  echo -n -e "Rscript outputgate.r $i 100000 3 10 3 C C D F F SG >> results.txt\n" >> makefile
done
