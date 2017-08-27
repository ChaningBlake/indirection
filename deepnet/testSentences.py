from sentenceSets import sentenceSets

#sentenceSets.standardGeneralization(3,10,200,100)
train_setX, train_setY, test_setX, test_setY = sentenceSets.standardGeneralization(3,10,200,100)
#train_setX, train_setY, test_setX, test_setY = sentenceSets.spuriousAnticorrelation(3,10,200,100)
#train_setX, train_setY, test_setX, test_setY = sentenceSets.fullCombinatorial(3,10,200,100)
#print(train_setX[0:1,])
#print(train_setY[0:1,])
#print(test_setX[0:1,])
#print(test_setY[0:1,])
