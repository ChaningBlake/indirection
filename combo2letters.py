#!/usr/bin/env python3
'''
October 17th 2019
@author: Blake Mullinax

This script reads the output of testSentences.r through stdin
and converts the combinations into letters to be used for training
an encoder/decoder.
'''
import sys

if len(sys.argv) != 3:
	print()
	print("Usage %s [# in training set, # in testing set]"%sys.argv[0])
	print()
	sys.exit(1)

letters = ['a','b','c','d','e',
           'f','g','h','i','j',
		   'k','l','m','n','o',
		   'p','q','r','s','t',
		   'u','v','w','x','y','z']
nTrain = int(sys.argv[1])
nTest = int(sys.argv[2])

# Read in each combination and output the coresponding
# letters
# (Only for training portion)
input()
for i in range(nTrain):
	line = input()
	line = line.split()
	for j in line[1:]:
		j = int(j)
		print(letters[j-1], end='')
	print()

# Do the same for testing portion
print()
input()
for i in range(nTest):
	line = input()
	line = line.split()
	for j in line[1:]:
		j = int(j)
		print(letters[j-1], end='')
	print()
