source('sentenceSets.r')
sets <- novelFiller(3,10,200,100)
print(sets$train)
print(sets$test)
