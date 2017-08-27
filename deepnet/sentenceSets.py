import numpy as np

class sentenceSets:

    #############################################################################
    ##
    ## This library creates sentences for the test sets used in the sentence role
    ## generalization tasks. All 'sets' returned are indices into the filler hrr matrix.
    ## It is assumed that roles are in order, 1 through n.
    ## This package assumes three roles and ten fillers, even though these are parameters.
    ## It also assumes train set size 200 and test set size 100
    ##
    ## 2017 Mike Jovanovich
    ##
    #############################################################################

    ## Creates one-hot vectors from class ints for the sentences
    ## Cols 1-nfillers are for fillers; roles are appended to the columns after fillers
    def oneHotEncode(cur_set,queries,answers,nroles,nfillers):
        ## Create the ndarray for the encoded vectors
        x = np.zeros(shape=(cur_set.shape[0],nroles+1,nfillers+nroles), dtype=float)
        y = np.zeros(shape=(cur_set.shape[0],nroles+1,nfillers), dtype=float)

        for sentence in range(cur_set.shape[0]):
            ## Add 'store' inputs
            for word in range(nroles):
                x[sentence,word,cur_set[sentence,word]] = y[sentence,word,cur_set[sentence,word]] = 1.0
                x[sentence,word,nfillers+word] =  1.0
            ## Add final 'retrieve' or 'query' input
            x[sentence,word+1,nfillers+queries[sentence]] = 1.0
            y[sentence,word+1,answers[sentence]] = 1.0
            
        return x, y

    ## This is a private function to check whether a sentence already exists in a 
    ## given set
    def setContains(s,cur_set):
        for i in range(cur_set.shape[0]):
            if np.array_equal(cur_set[i,],s):
                return True
        return False

    ## Training set is constrained to ensure that every word is presented as a filler
    ## in each of the roles. This protocol tests the ability of the network to adapt to
    ## arbitrary combinations of role/filler pairs, but not its ability to process novel
    ## role/filler pairs.
    def standardGeneralization(nroles,nfillers,size_train,size_test):

        ## What will be returned is a matrix where each row is a sentence
        trainX = np.zeros(shape=(size_train,nroles+1),dtype=int)
        testX = np.zeros(shape=(size_test,nroles+1),dtype=int)

        ## We try to make the number of times a word appears in each role
        times_word_in_role = int(size_train / nfillers / nroles)

        ## Get role query for final timestep and the answer that goes with it
        queries_train = np.random.randint(nroles,size=size_train)
        queries_test = np.random.randint(nroles,size=size_test)
        queries_train_coded = -1*queries_train-1
        queries_test_coded = -1*queries_test-1

        cur_row = 0
        for f in range(nfillers):
            for r in range(nroles):
                for t in range(times_word_in_role):
                    if cur_row <= size_train:

                        ## Make sure we put a unique sentence into the training set
                        sentence = np.random.choice(nfillers,nroles,replace=True)
                        sentence[r] = f
                        while( sentenceSets.setContains( sentence, trainX ) ):
                            sentence = np.random.choice(nfillers,nroles,replace=True)
                            sentence[r] = f

                        trainX[cur_row,] = np.append(sentence,queries_train_coded[cur_row])
                        cur_row = cur_row + 1

        ## Fill any extra non-uniform buffer at the end of the set with random sentences
        while cur_row < size_train:
            ## Make sure we put a unique sentence into the training set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentenceSets.setContains( sentence, trainX ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)

            trainX[cur_row,] = np.append(sentence,queries_train_coded[cur_row])
            cur_row = cur_row + 1

        ## Build the test set
        for i in range(size_test):
            ## Make sure we put a unique sentence into the test set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentenceSets.setContains( sentence, trainX ) 
              or sentenceSets.setContains( sentence, testX ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)
            testX[i,] = np.append(sentence,queries_test_coded[i])

        ## Get role query for final timestep and the answer that goes with it
        answers_train = np.asarray([trainX[i,queries_train[i]] for i in range(size_train)])
        answers_test = np.asarray([testX[i,queries_test[i]] for i in range(size_test)])

        trainY = np.zeros((size_train, nfillers))
        trainY[np.arange(size_train), answers_train] = 1
        testY = np.zeros((size_test, nfillers))
        testY[np.arange(size_test), answers_test] = 1

        ## debug
        #print(trainX[0:9,])
        #print(trainY[0:9,])
        #print(testX[0:9,])
        #print(testY[0:9,])

        return trainX, trainY, testX, testY


    ## Private function to check for anticorrelation in a sentence
    def containsAnticorrelation(sentence,anticorrelations):
        for w in range(len(sentence)):
            if anticorrelations[sentence[w]] in [x for i,x in enumerate(sentence) if i!=1]:
                return True
        return False

    ## In the training set certain pairs of words will never appear together 
    ## (they are then anticorrelated). In the test set these pairs of words will 
    ## appear together.
    def spuriousAnticorrelation(nroles,nfillers,size_train,size_test):

        ## Build the list of pairs of anticorrelated words
        ## Each word has one other that it cannot appear with
        anticorrelations = np.zeros(nfillers,dtype=int)
        for i in range(nfillers):
            if i % 2 == 0:
                anticorrelations[i] = i+1
            else:
                anticorrelations[i] = i-1

        ## What will be returned is a matrix where each row is a sentence
        trainX = np.zeros(shape=(size_train,nroles),dtype=int)
        testX = np.zeros(shape=(size_test,nroles),dtype=int)

        ## We try to make the number of times a word appears in each role
        times_word_in_role = int(size_train / nfillers / nroles)

        cur_row = 0
        for f in range(nfillers):
            for r in range(nroles):
                for t in range(times_word_in_role):
                    if cur_row <= size_train:

                        ## Make sure we put a unique sentence into the training set
                        sentence = np.random.choice(nfillers,nroles,replace=True)
                        sentence[r] = f
                        while( sentenceSets.containsAnticorrelation( sentence, anticorrelations ) 
                          or sentenceSets.setContains( sentence, trainX ) ):
                            sentence = np.random.choice(nfillers,nroles,replace=True)
                            sentence[r] = f

                        trainX[cur_row,] = sentence
                        cur_row = cur_row + 1

        ## Fill any extra non-uniform buffer at the end of the set with random sentences
        while cur_row < size_train:
            ## Make sure we put a unique sentence into the training set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentenceSets.containsAnticorrelation( sentence, anticorrelations ) 
              or sentenceSets.setContains( sentence, trainX ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)

            trainX[cur_row,] = sentence
            cur_row = cur_row + 1

        ## Build the test set
        for i in range(size_test ):
            ## Make sure we put a unique sentence into the test set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentenceSets.containsAnticorrelation( sentence, anticorrelations ) == False
              or sentenceSets.setContains( sentence, testX ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)
            testX[i,] = sentence

        ## Get role query for final timestep and the answer that goes with it
        queries_train = np.random.randint(nroles,size=size_train)
        answers_train = [trainX[i,queries_train[i]] for i in range(size_train)]
        queries_test = np.random.randint(nroles,size=size_test)
        answers_test = [testX[i,queries_test[i]] for i in range(size_test)]

        ## One-hot encode classes
        trainXX, trainXY = sentenceSets.oneHotEncode(trainX,queries_train,answers_train,nroles,nfillers)
        testXX, testXY = sentenceSets.oneHotEncode(testX,queries_test,answers_test,nroles,nfillers)

        return trainXX, trainXY, testXX, testXY

    ## Two of the ten fillers will never be used in role 'agent' (1) in 
    ## the training set. The test set will then present them in this role.
    def fullCombinatorial(nroles,nfillers,size_train,size_test):

        ## What will be returned is a matrix where each row is a sentence
        trainX = np.zeros(shape=(size_train,nroles),dtype=int)
        testX = np.zeros(shape=(size_test,nroles),dtype=int)

        ## We try to make the number of times a word appears in each role
        times_word_in_role = int(size_train / nfillers / nroles)

        cur_row = 0
        for f in range(nfillers):
            for r in range(nroles):
                if( r == 0 and (f == 0 or f == 1) ):
                    continue 
                for t in range(times_word_in_role):
                    if cur_row <= size_train:

                        ## Make sure we put a unique sentence into the training set
                        sentence = np.random.choice(nfillers,nroles,replace=True)
                        sentence[r] = f
                        while( sentence[0] == 0 or sentence[0] == 1 
                          or sentenceSets.setContains( sentence, trainX ) ):
                            sentence = np.random.choice(nfillers,nroles,replace=True)
                            sentence[r] = f

                        trainX[cur_row,] = sentence
                        cur_row = cur_row + 1

        ## Fill any extra non-uniform buffer at the end of the set with random sentences
        while cur_row < size_train:
            ## Make sure we put a unique sentence into the training set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentence[0] == 0 or sentence[0] == 1 
              or sentenceSets.setContains( sentence, trainX ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)

            trainX[cur_row,] = sentence
            cur_row = cur_row + 1

        ## Build the test set
        for i in range(size_test):
            ## Make sure we put a unique sentence into the test set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( (sentence[0] != 0 and sentence[0] != 1)
              or sentenceSets.setContains( sentence, testX ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)
            testX[i,] = sentence

        ## Get role query for final timestep and the answer that goes with it
        queries_train = np.random.randint(nroles,size=size_train)
        answers_train = [trainX[i,queries_train[i]] for i in range(size_train)]
        queries_test = np.random.randint(nroles,size=size_test)
        answers_test = [testX[i,queries_test[i]] for i in range(size_test)]

        ## One-hot encode classes
        trainXX, trainXY = sentenceSets.oneHotEncode(trainX,queries_train,answers_train,nroles,nfillers)
        testXX, testXY = sentenceSets.oneHotEncode(testX,queries_test,answers_test,nroles,nfillers)

        return trainXX, trainXY, testXX, testXY
