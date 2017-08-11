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
        train_set = np.zeros(shape=(size_train,nroles),dtype=int)
        test_set = np.zeros(shape=(size_test,nroles),dtype=int)

        ## We try to make the number of times a word appears in each role
        times_word_in_role = int(size_train / nfillers / nroles)

        cur_row = 1
        for f in range(nfillers):
            for r in range(nroles):
                for t in range(times_word_in_role):
                    if cur_row <= size_train:

                        ## Make sure we put a unique sentence into the training set
                        sentence = np.random.choice(nfillers,nroles,replace=True)
                        sentence[r] = f
                        while( sentenceSets.setContains( sentence, train_set ) ):
                            sentence = np.random.choice(nfillers,nroles,replace=True)
                            sentence[r] = f

                        train_set[cur_row,] = sentence
                        cur_row = cur_row + 1

        ## Fill any extra non-uniform buffer at the end of the set with random sentences
        while cur_row < size_train:
            ## Make sure we put a unique sentence into the training set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentenceSets.setContains( sentence, train_set ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)

            train_set[cur_row,] = sentence
            cur_row = cur_row + 1

        ## Build the test set
        for i in range(size_test ):
            ## Make sure we put a unique sentence into the test set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentenceSets.setContains( sentence, train_set ) 
              or sentenceSets.setContains( sentence, test_set ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)
            test_set[i,] = sentence

        return train_set, test_set


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
        train_set = np.zeros(shape=(size_train,nroles),dtype=int)
        test_set = np.zeros(shape=(size_test,nroles),dtype=int)

        ## We try to make the number of times a word appears in each role
        times_word_in_role = int(size_train / nfillers / nroles)

        cur_row = 1
        for f in range(nfillers):
            for r in range(nroles):
                for t in range(times_word_in_role):
                    if cur_row <= size_train:

                        ## Make sure we put a unique sentence into the training set
                        sentence = np.random.choice(nfillers,nroles,replace=True)
                        sentence[r] = f
                        while( sentenceSets.containsAnticorrelation( sentence, anticorrelations ) 
                          or sentenceSets.setContains( sentence, train_set ) ):
                            sentence = np.random.choice(nfillers,nroles,replace=True)
                            sentence[r] = f

                        train_set[cur_row,] = sentence
                        cur_row = cur_row + 1

        ## Fill any extra non-uniform buffer at the end of the set with random sentences
        while cur_row < size_train:
            ## Make sure we put a unique sentence into the training set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentenceSets.containsAnticorrelation( sentence, anticorrelations ) 
              or sentenceSets.setContains( sentence, train_set ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)

            train_set[cur_row,] = sentence
            cur_row = cur_row + 1

        ## Build the test set
        for i in range(size_test ):
            ## Make sure we put a unique sentence into the test set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentenceSets.containsAnticorrelation( sentence, anticorrelations ) == False
              or sentenceSets.setContains( sentence, test_set ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)
            test_set[i,] = sentence

        return train_set, test_set

    ## Two of the ten fillers will never be used in role 'agent' (1) in 
    ## the training set. The test set will then present them in this role.
    def fullCombinatorial(nroles,nfillers,size_train,size_test):

        ## What will be returned is a matrix where each row is a sentence
        train_set = np.zeros(shape=(size_train,nroles),dtype=int)
        test_set = np.zeros(shape=(size_test,nroles),dtype=int)

        ## We try to make the number of times a word appears in each role
        times_word_in_role = int(size_train / nfillers / nroles)

        cur_row = 1
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
                          or sentenceSets.setContains( sentence, train_set ) ):
                            sentence = np.random.choice(nfillers,nroles,replace=True)
                            sentence[r] = f

                        train_set[cur_row,] = sentence
                        cur_row = cur_row + 1

        ## Fill any extra non-uniform buffer at the end of the set with random sentences
        while cur_row < size_train:
            ## Make sure we put a unique sentence into the training set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( sentence[0] == 0 or sentence[0] == 1 
              or sentenceSets.setContains( sentence, train_set ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)

            train_set[cur_row,] = sentence
            cur_row = cur_row + 1

        ## Build the test set
        for i in range(size_test):
            ## Make sure we put a unique sentence into the test set
            sentence = np.random.choice(nfillers,nroles,replace=True)
            while( (sentence[0] != 0 and sentence[0] != 1)
              or sentenceSets.setContains( sentence, test_set ) ):
                sentence = np.random.choice(nfillers,nroles,replace=True)
            test_set[i,] = sentence

        return train_set, test_set
