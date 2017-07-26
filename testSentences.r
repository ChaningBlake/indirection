source('sentenceSets.r')
sets <- fullCombinatorial(3,10,200,100)
print(sets$test)
