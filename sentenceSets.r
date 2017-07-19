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
setContains <- function(s,set) {
    for( i in 1:dim(set)[1] ) {
        if( all(set[i,]==s) )
            return (TRUE)
    }
    return (FALSE)
}

## Training set is constrained to ensure that every word is presented as a filler
## in each of the roles. This protocol tests the ability of the network to adapt to
## arbitrary combinations of role/filler pairs, but not its ability to process novel
## role/filler pairs.
standardGeneralization <- function(nroles,nfillers,size_train,size_test) {

    ## What will be returned is a matrix where each row is a sentence
    train_set <- array(rep(-1,nroles*size_train),dim=c(size_train,nroles))
    test_set <- array(rep(-1,nroles*size_test),dim=c(size_test,nroles))

    ## We try to make the number of times a word appears in each role
    times_word_in_role <- as.integer(size_train / nfillers / nroles)

    cur_row <- 1
    for( f in 1:nfillers ) {
        for( r in 1:nroles ) {
            for( t in 1: times_word_in_role ) {
                if( cur_row <= size_train ) {
                    
                    ## Make sure we put a unique sentence into the training set
                    sentence <- sample(nfillers,nroles,replace=TRUE)
                    sentence[r] <- f
                    while( setContains( sentence, train_set ) ) {
                        sentence <- sample(nfillers,nroles,replace=TRUE)
                        sentence[r] <- f
                    }

                    train_set[cur_row,] <- sentence
                    cur_row <- cur_row + 1
                }
            }
        }
    }

    ## Fill any extra non-uniform buffer at the end of the set with random sentences
    while( cur_row <= size_train ) {
        ## Make sure we put a unique sentence into the training set
        sentence <- sample(nfillers,nroles,replace=TRUE)
        while( setContains( sentence, train_set ) ) {
            sentence <- sample(nfillers,nroles,replace=TRUE)
        }

        train_set[cur_row,] <- sentence
        cur_row <- cur_row + 1
    }

    ## Build the test set
    for( i in 1:size_test ) {
        ## Make sure we put a unique sentence into the test set
        sentence <- sample(nfillers,nroles,replace=TRUE)
        while( setContains( sentence, train_set ) || setContains( sentence, test_set ) ) {
            sentence <- sample(nfillers,nroles,replace=TRUE)
        }
        test_set[i,] <- sentence
    }

    ## Return the training and test sets
    return(list(
        train_set = train_set,
        test_set = test_set
    ))
}

## Private function to check for anticorrelation in a sentence
containsAnticorrelation <- function(sentence,anticorrelations) {
    for( w in 1:length(sentence) ) {
        if( anticorrelations[sentence[w]] %in% sentence[-w] )
            return (TRUE)
    }
    return (FALSE)
}

## In the training set certain pairs of words will never appear together 
## (they are then anticorrelated). In the test set these pairs of words will 
## appear together.
spuriousAnticorrelation <- function(nroles,nfillers,size_train,size_test) {

    ## Build the list of pairs of anticorrelated words
    ## Each word has one other that it cannot appear with
    anticorrelations <- rep(-1,nfillers)
    for( i in 1:nfillers ) {
        if( i %% 2 == 1 )
            anticorrelations[i] <- i+1
        else
            anticorrelations[i] <- i-1
    }

    ## What will be returned is a matrix where each row is a sentence
    train_set <- array(rep(-1,nroles*size_train),dim=c(size_train,nroles))
    test_set <- array(rep(-1,nroles*size_test),dim=c(size_test,nroles))

    ## We try to make the number of times a word appears in each role
    times_word_in_role <- as.integer(size_train / nfillers / nroles)

    cur_row <- 1
    for( f in 1:nfillers ) {
        for( r in 1:nroles ) {
            for( t in 1: times_word_in_role ) {
                if( cur_row <= size_train ) {
                    
                    ## Make sure we put a unique sentence into the training set
                    sentence <- sample((1:nfillers)[-anticorrelations[f]],nroles,replace=TRUE)
                    sentence[r] <- f
                    while( setContains( sentence, train_set ) ) {
                        sentence <- sample((1:nfillers)[-anticorrelations[f]],nroles,replace=TRUE)
                        sentence[r] <- f
                    }

                    train_set[cur_row,] <- sentence
                    cur_row <- cur_row + 1
                }
            }
        }
    }

    ## Fill any extra non-uniform buffer at the end of the set with random sentences
    while( cur_row <= size_train ) {
        ## Make sure we put a unique sentence into the training set
        sentence <- sample(nfillers,nroles,replace=TRUE)
        while( setContains( sentence, train_set ) || containsAnticorrelation( sentence, anticorrelations ) ) {
            sentence <- sample(nfillers,nroles,replace=TRUE)
        }
    
        train_set[cur_row,] <- sentence
        cur_row <- cur_row + 1
    }

    ## Build the test set
    for( i in 1:size_test ) {
        ## Make sure we put a unique sentence into the test set
        sentence <- sample(nfillers,nroles,replace=TRUE)
        while( !containsAnticorrelation( sentence, anticorrelations ) || setContains( sentence, test_set ) ) {
            sentence <- sample(nfillers,nroles,replace=TRUE)
        }
        test_set[i,] <- sentence
    }

    ## Return the training and test sets
    return(list(
        train_set = train_set,
        test_set = test_set
    ))
}

## Two of the ten fillers will never be used in role 'agent' (1) in 
## the training set. The test set will then present them in this role.
fullCombinatorial <- function(nroles,nfillers,size_train,size_test) {

    ## What will be returned is a matrix where each row is a sentence
    train_set <- array(rep(-1,nroles*size_train),dim=c(size_train,nroles))
    test_set <- array(rep(-1,nroles*size_test),dim=c(size_test,nroles))

    ## We try to make the number of times a word appears in each role
    times_word_in_role <- as.integer(size_train / nfillers / nroles)

    cur_row <- 1
    for( f in 1:nfillers ) {
        for( r in 1:nroles ) {
            if( r == 1 && (f == 1 || f == 2) )
                next
            for( t in 1: times_word_in_role ) {
                if( cur_row <= size_train ) {
                    
                    ## Make sure we put a unique sentence into the training set
                    sentence <- sample(nfillers,nroles,replace=TRUE)
                    sentence[r] <- f
                    while( sentence[1] == 1 || sentence[1] == 2 || setContains( sentence, train_set ) ) {
                        sentence <- sample(nfillers,nroles,replace=TRUE)
                        sentence[r] <- f
                    }

                    train_set[cur_row,] <- sentence
                    cur_row <- cur_row + 1
                }
            }
        }
    }

    ## Fill any extra non-uniform buffer at the end of the set with random sentences
    while( cur_row <= size_train ) {
        ## Make sure we put a unique sentence into the training set
        sentence <- sample(nfillers,nroles,replace=TRUE)
        while( sentence[1] == 1 || sentence[1] == 2 || setContains( sentence, train_set ) ) {
            sentence <- sample(nfillers,nroles,replace=TRUE)
        }

        train_set[cur_row,] <- sentence
        cur_row <- cur_row + 1
    }

    ## Build the test set
    for( i in 1:size_test ) {
        ## Make sure we put a unique sentence into the test set
        sentence <- sample(nfillers,nroles,replace=TRUE)
        while( (sentence[1] != 1 && sentence[1] != 2) || setContains( sentence, test_set ) ) {
            sentence <- sample(nfillers,nroles,replace=TRUE)
        }
        test_set[i,] <- sentence
    }

    ## Return the training and test sets
    return(list(
        train_set = train_set,
        test_set = test_set
    ))
}
