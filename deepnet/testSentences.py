from sentenceSets import sentenceSets
#[train_set,test_set] = sentenceSets.standardGeneralization(3,10,200,100)
#[train_set,test_set] = sentenceSets.spuriousAnticorrelation(3,10,200,100)
[train_set,test_set] = sentenceSets.fullCombinatorial(3,10,200,100)
print(train_set[1:10,])
print(test_set[1:10,])
