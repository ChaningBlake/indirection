#############################################################################
##
## Indirection - phase 2: output gating 
## Role/filler recall task
## 2017 Mike Jovanovich
##
## In this version OG sees WM after it has been update by IG within a timestep.
## The action selection if this version simply checks if WM_out outputs only
## the correct filler
##
#############################################################################

## Usage
# Rscript outputgate.r 1 100000 3 10 3 C C D F F SG

#############################################################################
## Parameter initialization
#############################################################################

source('hrr.r')
source('sentenceSets.r')
require(methods)
require(Matrix)

## Build args dictionary
argnames <- list('seed','ntasks','nstripes','nfillers','nroles','state_cd','sid_cd','interstripe_cd',
  'use_sids_input','use_sids_output','gen_test')
args <- commandArgs(trailingOnly = TRUE)

## Set the seed for repeatability
set.seed(as.integer(args[which(argnames=='seed')]))

## Length of the HRRs
n <- 1024
## Number of possible fillers
nfillers <- as.integer(args[which(argnames=='nfillers')])
## Number of possible roles
nroles <- as.integer(args[which(argnames=='nroles')])
## Number of WM stripes
nstripes <- as.integer(args[which(argnames=='nstripes')])

## Define conj/disj scheme
state_cd <- args[which(argnames=='state_cd')]
sid_cd <- args[which(argnames=='sid_cd')]
interstripe_cd <- args[which(argnames=='interstripe_cd')]

## Identity vectors
hrr_z <- rep(0,n)
hrr_o <- rep(0,n)
hrr_o[1] <- 1

# Set this according to conj. disj. scheme;
if( interstripe_cd == 'D' ) {
    hrr_i <- hrr_z 
} else {
    hrr_i <- hrr_o
}

## Filler vectors 
## Fillers don't have friendly names, only indexes
## To index: [,filler]
fillers <- replicate(nfillers,hrr(n,normalized=TRUE))
wm_fillers <- replicate(nfillers,hrr(n,normalized=TRUE))
f_fillers <- paste('f',1:nfillers,sep='') # Friendly names

## Role vectors 
## Roles don't have friendly names, only indexes
## To index: [,role]
roles <- replicate(nroles,hrr(n,normalized=TRUE))

## Stripe ID (SID) vectors 
## To index: [,sid]
sids <- replicate(nstripes,hrr(n,normalized=TRUE))
use_sids_input <- args[which(argnames=='use_sids_input')] == 'T'
use_sids_output <- args[which(argnames=='use_sids_output')] == 'T'

## Op code vectors
## To index: [,1] for store, [,2] for retrieve
ops <- replicate(2,hrr(n,normalized=TRUE))

## Op code vectors
## To index: [,1] go, [,2] no go 
gono <- replicate(2,hrr(n,normalized=TRUE))

## TD weight vectors
## A suffix of '_m' inicates relation to the maintenance layer NN
## To index: [,stripe]
W_m <- replicate(nstripes,hrr(n,normalized=TRUE))
W_o <- replicate(nstripes,hrr(n,normalized=TRUE))

## Action weight vectors
## There are 'nfillers' output units for this network
## Each output unit should come to represent the desired filler
## To index: [,output_unit]
W_a <- replicate(nfillers,hrr(n,normalized=TRUE))

## TD parameters
default_reward <- 0.0
success_reward <- 1.0
bias_m <- 1.0
bias_o <- 1.0
bias_a <- 1.0
gamma_m <- 1.0
gamma_o <- 1.0
gamma_a <- 1.0
lambda_m <- 0.9
lambda_o <- 0.9
lrate_m <- 0.1
lrate_o <- 0.1
lrate_a <- 0.1
epsilon_m <- .025
epsilon_o <- .025

## Task parameters
max_tasks <- as.integer(args[which(argnames=='ntasks')])
cur_task <- 1
cur_block_task <- 1
block_tasks_correct <- 0

## Get training and test sets
train_set_size <- 200
test_set_size <- 100

## Choose testing protocol
if( args[which(argnames=='gen_test')] == 'SG' ) {
    sets <- standardGeneralization(nroles,nfillers,train_set_size,test_set_size)
} else if( args[which(argnames=='gen_test')] == 'SA' ) {
    sets <- spuriousAnticorrelation(nroles,nfillers,train_set_size,test_set_size)
} else if( args[which(argnames=='gen_test')] == 'FC' ) {
    sets <- fullCombinatorial(nroles,nfillers,train_set_size,test_set_size)
} else if( args[which(argnames=='gen_test')] == 'NF' ) {
    sets <- novelFiller(nroles,nfillers,train_set_size,test_set_size)
}

#############################################################################
## selectAction:
##
## This function handles action selection. To keep things simple, there must
## be only one WM_out filler, otherwise a -1 action is returned.
##
#############################################################################
selectAction <- function() {
    action <- -1
    for( i in 1:nstripes ) {
        if( f_stripes_o[i] != 'I' )
            ## if there are more than one fillers return -1
            if( action != -1 )
                return(-1)
            else
                action <- as.integer(substr(f_stripes_o[i],2,nchar(f_stripes_o[i])))
    }
    return (action)
}

#############################################################################
## getState: Returns role/op combo
#############################################################################
getState <- function(o,r) {
    ## Encode state (role,op)
    if( state_cd == 'C' ) {
        state <- convolve(ops[,o],roles[,r])
    } else {
        state <- cnorm(ops[,o] + roles[,r])
    }
    #state <- roles[,r]
    return (state)
}

#############################################################################
## inputGate:
##
## This function handles input gating for a single timestep/operation in the task.
## At each timestep a stripe can either retain its contents or update with the 
## provided filler. 
##
## Multiple stripes update in a single timestep.
##
#############################################################################
inputGate <- function(state,f=-1) {

    ## Get NN output unit values
    ## Note that state_wm is for the previous timestep
    ## We will return this to use in the eligibility trace
    #state_wm <- convolve(state,cur_wm_m)

    ## This is state_wm convolved with both the go and no go hrrs
    #state_wm_gono <- apply(gono,2,convolve,state_wm)
    state_gono <- apply(gono,2,convolve,state)

    ## Build to matrix of eligility trace vectors that will be returned
    ## for TD updates; default to no go
    #elig <- replicate(nstripes,state_wm_gono[,2])
    elig <- replicate(nstripes,state_gono[,2])
    vals <- rep(0.0,nstripes)
    open <- rep(FALSE,nstripes)

    ## Determine if open or close value is better
    for( i in 1:nstripes ) {
        #temp_vals <- (apply(state_wm_gono,2,nndot,W_m[,i]) + bias_m)
        temp_vals <- (apply(state_gono,2,nndot,W_m[,i]) + bias_m)
        vals[i] <- max(temp_vals)

        ## Epsilon soft policy
        r <- runif(1)
        if( r < epsilon_m ) {
            ## Pick a random open or close state
            r <- runif(1)
            if( r > .5 ) {
                #elig[,i] <- state_wm_gono[,1]
                elig[,i] <- state_gono[,1]
                open[i] <- TRUE
                vals[i] <- temp_vals[1]
            } else {
                vals[i] <- temp_vals[2]
            }
        } else if( temp_vals[1] > temp_vals[2] ) {
            #elig[,i] <- state_wm_gono[,1]
            elig[,i] <- state_gono[,1]
            open[i] <- TRUE
        }
    }


    ## Update WM_m contents 
    ## We are convolving the fill with the appropriate SID
    for( i in 1:nstripes ) {
        if( open[i] ) {
            if( f == -1 ) {
                stripes_m[,i] <- hrr_i
                stripes_mo[,i] <- hrr_i
                f_stripes_m[i] <- 'I'
            } else {
                if( sid_cd == 'C' ) {
                    if( use_sids_input )
                        stripes_m[,i] <- convolve(fillers[,f],sids[,i])
                    if( use_sids_output )
                        stripes_mo[,i] <- convolve(fillers[,f],sids[,i])
                } else {
                    if( use_sids_input )
                        stripes_m[,i] <- cnorm(fillers[,f]+sids[,i])
                    if( use_sids_output )
                        stripes_mo[,i] <- cnorm(fillers[,f]+sids[,i])
                }
                if( !use_sids_input )
                    stripes_m[,i] <- fillers[,f]
                if( !use_sids_output )
                    stripes_mo[,i] <- fillers[,f]
                f_stripes_m[i] <- f_fillers[f]
            }
        }
    }

    ## Get updated wm representation
    if( interstripe_cd == 'D' ) {
        wm <- cnorm(apply(stripes_m,1,sum))
        if( is.nan(wm[1]) )
            wm <- hrr_o
    } else {
        wm <- mconvolve(stripes_m)
    }

    return(list(
        wm = wm,
        elig = elig,
        vals = vals,
        stripes_m = stripes_m,
        stripes_mo = stripes_mo,
        f_stripes_m = f_stripes_m
    ))
}

#############################################################################
## outputGate:
##
## This function handles output gating for a single timestep/operation in the task.
## Each PFC stripe can either output its contents or not.
##
#############################################################################
outputGate <- function(state) {

    ## Start with a blank slate
    stripes_o <- replicate(nstripes,hrr_i)
    f_stripes_o <- replicate(nstripes,'I')

    ## Get NN output unit values
    ## Note that state_wm is for the previous timestep
    ## We will return this to use in the eligibility trace
    #state_wm <- convolve(state,cur_wm_m)

    ## This is state_wm convolved with both the go and no go hrrs
    #state_wm_gono <- apply(gono,2,convolve,state_wm)
    state_gono <- apply(gono,2,convolve,state)

    ## Build to matrix of eligility trace vectors that will be returned
    ## for TD updates; default to no go
    #elig <- replicate(nstripes,state_wm_gono[,2])
    elig <- replicate(nstripes,state_gono[,2])
    vals <- rep(0.0,nstripes)
    open <- rep(FALSE,nstripes)

    ## Determine if open or close value is better
    for( i in 1:nstripes ) {

        #temp_vals <- (apply(state_wm_gono,2,nndot,W_o[,i]) + bias_o)
        temp_vals <- (apply(state_gono,2,nndot,W_o[,i]) + bias_o)
        vals[i] <- max(temp_vals)

        ## Epsilon soft policy
        r <- runif(1)
        if( r < epsilon_o ) {
            ## Pick a random open or close state
            r <- runif(1)
            if( r > .5 ) {
                #elig[,i] <- state_wm_gono[,1]
                elig[,i] <- state_gono[,1]
                open[i] <- TRUE
                vals[i] <- temp_vals[1]
            } else {
                vals[i] <- temp_vals[2]
            }
        } else if( temp_vals[1] > temp_vals[2] ) {
            #elig[,i] <- state_wm_gono[,1]
            elig[,i] <- state_gono[,1]
            open[i] <- TRUE
        }
    }


    ## Update WM_o contents 
    ## We are convolving the fill with the appropriate SID
    for( i in 1:nstripes ) {
        if( open[i] ) {
            stripes_o[,i] <- stripes_mo[,i]
            f_stripes_o[i] <- f_stripes_m[i]
        }
    }

    ## Get updated wm representation
    if( interstripe_cd == 'D' ) {
        wm <- cnorm(apply(stripes_o,1,sum))
        if( is.nan(wm[1]) )
            ## Change this depending on how we want 'nothing' to look for action selection
            #wm <- hrr_o
            wm <- hrr_z
    } else {
        wm <- mconvolve(stripes_o)
    }

    return(list(
        wm = wm,
        elig = elig,
        vals = vals,
        f_stripes_o = f_stripes_o
    ))
}

## Continue until we hit the max_tasks, or we have a block success rate of 95%
while( cur_task <= max_tasks ) {

    ## Setup 200 task blocks
    if( cur_task %% 200 == 1 ) {
        if( block_tasks_correct/200 >= .95 )
            break
        cur_block_task <- 1
        block_tasks_correct <- 0
    }

    #############################################################################
    ## Initialization and task setup
    #############################################################################

    reward <- default_reward
    elig_m <- replicate(nstripes,rep(0,n)) # There is an elig. trace for each BG input gate
    elig_o <- replicate(nstripes,rep(0,n)) # There is an elig. trace for each BG output gate
    prev_val_m <- rep(0.0,nstripes)     # prev. timestep values for maintenance NN output layer units
    prev_val_o <- rep(0.0,nstripes)     # prev. timestep values for output NN output layer units
    cur_wm_m <- hrr_o
    cur_wm_o <- hrr_o

    ## Fill stripes with identity vector
    stripes_m <- replicate(nstripes,hrr_i)  # Input layer stripes (hrrs)
    stripes_mo <- replicate(nstripes,hrr_i)  # Output layer stripes (hrrs); these are same as above, but with no SID
    f_stripes_m <- replicate(nstripes,'I')  # Friendly name for stripe contents of input WM layer
    f_stripes_o <- replicate(nstripes,'I')  # Friendly name for stripe contents of output WM layer

    #############################################################################
    ## Store fillers
    #############################################################################

    ## Choose a sentence at random from the sample set
    s_f <- sets$train_set[sample(train_set_size,1),]

    ## Permute the sentence so that roles are not always presented in the same order
    p <- sample(nroles,nroles,replace=FALSE)

    for( t in 1:nroles ) {
        #cat(sprintf('t=%d\n',t)) #debug

        state <- getState(1,p[t])

        #############################################################################
        ## Input gating
        #############################################################################

        ## Update WM input layer global variables
        ig <- inputGate(state,s_f[p[t]])
        cur_wm_m <- ig$wm
        stripes_m <- ig$stripes_m
        stripes_mo <- ig$stripes_mo
        f_stripes_m <- ig$f_stripes_m

        #############################################################################
        ## Output gating
        #############################################################################

        ## Update WM output layer global variables
        og <- outputGate(state)
        cur_wm_o <- og$wm
        f_stripes_o <- og$f_stripes_o

        #############################################################################
        ## Neural network and TD training for this trial
        #############################################################################

        ## INPUT GATE
        error <- (reward + gamma_m * ig$vals) - prev_val_m
        for( i in 1:nstripes ) {
            W_m[,i] <- W_m[,i] + lrate_m * error[i] * elig_m[,i]
            elig_m[,i] <- cnorm(lambda_m * elig_m[,i] + ig$elig[,i])
        }
        prev_val_m <- ig$vals 

        ## OUTPUT GATE
        error <- (reward + gamma_o * og$vals) - prev_val_o
        for( i in 1:nstripes ) {
            W_o[,i] <- W_o[,i] + lrate_o * error[i] * elig_o[,i]
            elig_o[,i] <- cnorm(lambda_o * elig_o[,i] + og$elig[,i])
        }
        prev_val_o <- og$vals 

    }
    #cat(sprintf('t=%d\n',t+1)) #debug

    #############################################################################
    ## Query for roles
    #############################################################################

    ## Permute the sentence so that roles are not always queried in the same order
    nqueries <- 1
    p <- sample(nroles,nqueries,replace=FALSE)
    correct_trial <- rep(0,nqueries)

    for( t in 1:nqueries ) {
        state <- getState(2,p[t])

        #############################################################################
        ## Input gating
        #############################################################################

        ## Update WM input layer global variables
        ig <- inputGate(state)
        cur_wm_m <- ig$wm
        stripes_m <- ig$stripes_m
        stripes_mo <- ig$stripes_mo
        f_stripes_m <- ig$f_stripes_m

        #############################################################################
        ## Output gating
        #############################################################################

        ## Update WM output layer global variables
        og <- outputGate(state)
        cur_wm_o <- og$wm
        f_stripes_o <- og$f_stripes_o

        #############################################################################
        ## Action selection
        #############################################################################
        
        ac <- selectAction()

        #############################################################################
        ## Neural network and TD training for this trial
        #############################################################################

        ## INPUT GATE
        error <- (reward + gamma_m * ig$vals) - prev_val_m
        for( i in 1:nstripes ) {
            W_m[,i] <- W_m[,i] + lrate_m * error[i] * elig_m[,i]
            elig_m[,i] <- cnorm(lambda_m * elig_m[,i] + ig$elig[,i])
        }
        prev_val_m <- ig$vals 

        ## OUTPUT GATE
        error <- (reward + gamma_o * og$vals) - prev_val_o
        for( i in 1:nstripes ) {
            W_o[,i] <- W_o[,i] + lrate_o * error[i] * elig_o[,i]
            elig_o[,i] <- cnorm(lambda_o * elig_o[,i] + og$elig[,i])
        }
        prev_val_o <- og$vals 

        ## Determine correctness
        ## The trial is correct of the selected action matches the filler that
        ## was paired with the requested role.
        if( ac == s_f[p[t]] )
            correct_trial[t] <- 1

        #############################################################################
        ## Absorb reward
        #############################################################################
        if( t == nqueries ) {

            ## Reward if entire sequence is correct
            ## Update block correct tally
            if( sum(correct_trial) == nqueries ) {
                block_tasks_correct <- block_tasks_correct + 1
                reward <- success_reward
            } else {
                reward <- default_reward
            }

            ## Input NN
            error <- reward - ig$vals 
            for( i in 1:nstripes ) {
                W_m[,i] <- W_m[,i] + lrate_m * error[i] * elig_m[,i]
            }

            ## Output NN
            error <- reward - og$vals 
            for( i in 1:nstripes ) {
                W_o[,i] <- W_o[,i] + lrate_o * error[i] * elig_o[,i]
            }
        }
    }

    #############################################################################
    ## Output prints
    #############################################################################

    #if( FALSE ) {
    if( cur_task %% 200 == 0 ) {
        cat(sprintf('Tasks Complete: %d\n',cur_task))
        cat(sprintf('Block Accuracy: %.2f\n',(block_tasks_correct/200)*100))

        ## Only printing final request state here
        cat('Input WM Layer: \t')
        cat(f_stripes_m)
        cat('\n')
        cat('Output WM Layer: \t')
        cat(f_stripes_o)
        cat('\n')
        cat(sprintf('Requested Role: %d\n',p[t]))
        cat(sprintf('Correct Action: %d\n',s_f[p[t]]))
        cat('\n')
    }

    #############################################################################
    ## Task wrapup
    #############################################################################

    ## Increment task tally
    cur_task <- cur_task + 1
}

## Print final results
#cat(sprintf('%d\n',cur_task))
cat(sprintf('Final Block Accuracy: %.2f\n',(block_tasks_correct/200)*100))

#############################################################################
## Generalization Test
#############################################################################
if(TRUE) {
    novel_tasks_correct <- 0
    for( i in 1:test_set_size ) {
        correct_trial <- rep(0,nqueries)

        #############################################################################
        ## Store fillers
        #############################################################################

        ## Retrieve novel sentence from the test set
        ## We aren't training here so no need to permute
        s_f <- sets$test_set[i,]

        ## Permute the sentence so that roles are not always presented in the same order
        p <- sample(nroles,nroles,replace=FALSE)

        ## Do a 'Store' for each filler
        ## Action selection can be skipped here
        ## Do not do any training
        for( t in 1:nroles ) {
            state <- getState(1,p[t])

            ig <- inputGate(state,s_f[p[t]])
            cur_wm_m <- ig$wm
            stripes_m <- ig$stripes_m
            stripes_mo <- ig$stripes_mo
            f_stripes_m <- ig$f_stripes_m

            og <- outputGate(state)
            cur_wm_o <- og$wm
            f_stripes_o <- og$f_stripes_o
        }

        ## Do a 'Retrieve' input gate, output gate, and select action
        p <- sample(nroles,nqueries,replace=FALSE)
        for( t in 1:nqueries ) {

            ## We can only query role 1 for the FC test
            ## consequently, nqueries must be set to 1
            state <- if(args[which(argnames=='gen_test')] == 'FC') getState(2,1) else getState(2,p[t])
            answer <- if(args[which(argnames=='gen_test')] == 'FC') s_f[1] else s_f[p[t]] 

            ig <- inputGate(state)
            cur_wm_m <- ig$wm
            stripes_m <- ig$stripes_m
            stripes_mo <- ig$stripes_mo
            f_stripes_m <- ig$f_stripes_m

            og <- outputGate(state)
            cur_wm_o <- og$wm
            f_stripes_o <- og$f_stripes_o

            ac <- selectAction()
            if( ac == answer )
                correct_trial[t] <- 1
        }

        ## Determine if entire sequence is correct
        ## Not sure if we'll want to modify protocol above to train for this
        if( sum(correct_trial) == nqueries )
            novel_tasks_correct <- novel_tasks_correct + 1
    }

    ## Print final results
    cat(sprintf('Generalization Accuracy: %d\n',novel_tasks_correct))
    #cat(sprintf('%d\n',novel_tasks_correct))
} ## End generalization test
