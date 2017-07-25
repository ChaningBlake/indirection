#############################################################################
##
## Indirection - phase 1: input gating 
## Role/filler recall task
## 2017 Mike Jovanovich
##
## Actions must be learned for this model, since there is no output gating.
## The signal to drive action choice is all of WM, along with a prompt for the role.
##
#############################################################################

## Usage
# Rscript inputgate.r 1 100000 3 10 3 C C D SG

#############################################################################
## Parameter initialization
#############################################################################

source('hrr.r')
source('sentenceSets.r')
require(methods)
require(Matrix)

## Build args dictionary
argnames <- list('seed','ntasks','nstripes','nfillers','nroles','state_cd','sid_cd','interstripe_cd','gen_test')
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
interstripe_cd <- args[which(argnames=='interstripe_cd')]
sid_cd <- args[which(argnames=='sid_cd')]

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

## Action weight vectors
## There are 'nfillers' output units for this network
## Each output unit should come to represent the desired filler
## To index: [,output_unit]
W_a <- replicate(nfillers,hrr(n,normalized=TRUE))

## TD parameters
default_reward <- 0.0
success_reward <- 1.0
bias_m <- 1.0
bias_a <- 1.0
gamma_m <- 1.0
gamma_a <- 1.0
lambda_m <- 0.9
lrate_m <- 0.1
lrate_a <- 0.1
ac_softmax_t <- .125    ## Higher temperatures will increase chance of random action
epsilon_m <- .025

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
}

## All sentences from the training and test sets are initially in order by role
s_r <- 1:nroles

#############################################################################
## softmax: The softmax function
#############################################################################
softmax <- function(x) {
    return ((exp(x/ac_softmax_t)) / sum(exp(x/ac_softmax_t)))
}

#############################################################################
## softmax_select: Returns a value based on softmax probablities
#############################################################################
softmax_select <- function(x) {
    ## Get softmax vals
    soft_vals <- softmax(x)

    ## Select action
    r <- runif(1)
    upper_bound <- 0.0
    for( i in 1:length(x) ) {
        upper_bound <- upper_bound + soft_vals[i]
        if( r <= upper_bound )
            return (i)
    }

    ## This should never happen; softmax should add to one
    return (-1)
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
inputGate <- function(o,r,f=-1) {

    ## Encode state (role,op)
    if( state_cd == 'C' ) {
        state <- convolve(ops[,o],roles[,r])
    } else {
        state <- cnorm(ops[,o] + roles[,r])
    }

    ## Get NN output unit values
    ## Note that state_wm is for the previous timestep
    ## We will return this to use in the eligibility trace
    state_wm <- convolve(state,cur_wm)

    ## This is state_wm convolved with both the go and no go hrrs
    state_wm_gono <- apply(gono,2,convolve,state_wm)

    ## Build to matrix of eligility trace vectors that will be returned
    ## for TD updates; default to no go
    elig <- replicate(nstripes,state_wm_gono[,2])
    vals <- rep(0.0,nstripes)
    open <- rep(FALSE,nstripes)

    ## Determine if open or close value is better
    for( i in 1:nstripes ) {
        temp_vals <- (apply(state_wm_gono,2,nndot,W_m[,i]) + bias_m)
        vals[i] <- max(temp_vals)

        r <- runif(1)
        ## Epsilon soft policy
        if( r < epsilon_m ) {
            ## Pick a random open or close state
            r <- runif(1)
            if( r > .5 ) {
                elig[,i] <- state_wm_gono[,1]
                open[i] <- TRUE
                vals[i] <- temp_vals[1]
            } else {
                vals[i] <- temp_vals[2]
            }
        } else if( temp_vals[1] > temp_vals[2] ) {
            elig[,i] <- state_wm_gono[,1]
            open[i] <- TRUE
        }
    }


    ## Update WM_m contents 
    ## We are convolving the fill with the appropriate SID
    for( i in 1:nstripes ) {
        if( open[i] ) {
            if( f == -1 ) {
                stripes_m[,i] <- hrr_i
                f_stripes_m[i] <- 'I'
            } else {
                if( sid_cd == 'C' ) {
                    stripes_m[,i] <- convolve(fillers[,f],sids[,i])
                } else {
                    stripes_m[,i] <- cnorm(fillers[,f]+sids[,i])
                }
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
        #state = state,
        wm = wm,
        elig = elig,
        vals = vals,
        stripes_m = stripes_m,
        f_stripes_m = f_stripes_m
    ))
}

#############################################################################
## selectAction:
##
## This function handles action selection. The hidden layer takes as input
## working memory contents along with the role that is being queried.
## The max output unit is used to determine trial correctness.
##
#############################################################################
selectAction <- function(r) {

    ## Get NN output and output unit (action) with max value
    wm_role <- convolve(roles[,r],cur_wm)
    output <- (apply(W_a,2,nndot,wm_role) + bias_a)
    action <- softmax_select(output)
    
    return(list(
        wm_role = wm_role,
        val = output,
        action = action
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

    elig_m <- replicate(nstripes,rep(0,n)) # There is an elig. trace for each BG input gate
    prev_val_m <- rep(0.0,nstripes)     # prev. timestep values for maintenance NN output layer units
    reward <- default_reward
    cur_wm <- hrr_o

    ## Fill stripes with identity vector
    stripes_m <- replicate(nstripes,hrr_i)  # Input layer stripes (hrrs)
    f_stripes_m <- replicate(nstripes,'I')  # Friendly name for stripe contents of input WM layer

    #############################################################################
    ## Store fillers
    #############################################################################

    ## Choose a sentence at random from the sample set
    s_f <- sets$train_set[sample(train_set_size,1),]

    ## Permute the sentence so that roles are not always presented in the same order
    p <- sample(nroles,nroles,replace=FALSE)

    for( t in 1:nroles ) {
        #cat(sprintf('t=%d\n',t))

        #############################################################################
        ## Input gating
        #############################################################################

        ## Update WM input layer global variables
        ig <- inputGate(1,s_r[p[t]],s_f[p[t]])
        cur_wm <- ig$wm
        stripes_m <- ig$stripes_m
        f_stripes_m <- ig$f_stripes_m

        #############################################################################
        ## Action selection
        #############################################################################
        
        ## Success reward is given if the correct output unit is chosen
        ac <- selectAction(s_r[p[t]])
        wm_role <- ac$wm_role

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

        ## ACTION SELECTION
        ## This is not TD learning - just a NN
        answer <- rep(0,nfillers)
        answer[s_f[p[t]]] <- 1
        error <- answer - ac$val
        W_a <- (W_a + lrate_a * t(error %*% t(wm_role)))
    }
    #cat(sprintf('t=%d\n',t+1))

    #############################################################################
    ## Query for roles
    #############################################################################

    ## Permute the sentence so that roles are not always queried in the same order
    nqueries <- 1
    p <- sample(nroles,nqueries,replace=FALSE)
    correct_trial <- rep(0,nqueries)

    for( t in 1:nqueries ) {

        #############################################################################
        ## Input gating
        #############################################################################

        ## Update WM input layer global variables
        #ig <- inputGate(2,s_r[req])
        #cur_wm <- ig$wm
        #stripes_m <- ig$stripes_m
        #f_stripes_m <- ig$f_stripes_m

        #############################################################################
        ## Action selection
        #############################################################################
        
        ## Success reward is given if the correct output unit is chosen
        ac <- selectAction(s_r[p[t]])
        wm_role <- ac$wm_role

        #############################################################################
        ## Neural network and TD training for this trial
        #############################################################################

        ## INPUT GATE
        #error <- (reward + gamma_m * ig$vals) - prev_val_m
        #for( i in 1:nstripes ) {
            #W_m[,i] <- W_m[,i] + lrate_m * error[i] * elig_m[,i]
            #elig_m[,i] <- cnorm(lambda_m * elig_m[,i] + ig$elig[,i])
        #}
        #prev_val_m <- ig$vals 

        ## ACTION SELECTION
        ## This is not TD learning - just a NN
        answer <- rep(0,nfillers)
        answer[s_f[p[t]]] <- 1
        error <- answer - ac$val 
        W_a <- (W_a + lrate_a * t(error %*% t(wm_role)))

        ## Determine trial correctness.
        ## The trial is correct of the selected action matches the filler that
        ## was paired with the requested role.
        if( ac$action == s_f[p[t]] )
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
        cat(round(softmax(ac$val),4))
        cat('\n')
        cat(sprintf('Requested Role: %d\n',p[t]))
        cat(sprintf('Correct Action: %d\n',s_f[p[t]]))
        cat('\n')
    }

    if( FALSE ) {
    #if( cur_task %% 200 == 0 ) {
        ## For each stripe output open and close values
        for( s in 1:nstripes ) {
            for( r in 1:nroles ) {
                for( o in 1:2 ) {
                    if( state_cd == 'C' ) {
                        state_wm <- convolve(convolve(ops[,o],roles[,r]),cur_wm)
                    } else {
                        state_wm <- convolve(cnorm(ops[,o] + roles[,r]),cur_wm)
                    }
                    for( g in 1:2 ) {
                        cat(sprintf('%.2f',nndot(W_m[,s],convolve(state_wm,gono[,g]))+bias_m))
                        if( s != nstripes || o != 2 || r != nroles || g != 2 )
                            cat(',')
                    }
                }
            }
        }
        cat('\n')
    }

    #############################################################################
    ## Task wrapup
    #############################################################################

    ## Increment task tally
    cur_task <- cur_task + 1
}

## Print final results
cat(sprintf('%d\n',cur_task))
#cat(sprintf('Final Block Accuracy: %.2f\n',(block_tasks_correct/200)*100))

#############################################################################
## Generalization Test
#############################################################################
if(FALSE) {
    novel_tasks_correct <- 0
    for( i in 1:test_set_size ) {
        correct_trial <- rep(0,nqueries)

        #############################################################################
        ## Store fillers
        #############################################################################

        ## Retrieve novel sentence from the test set
        ## We aren't training here so no need to permute
        s_f <- sets$test_set[i,]

        ## Do a 'Store' for each filler
        ## Action selection can be skipped here
        ## Do not do any training
        for( t in 1:nroles ) {
            ## Input Gate
            ig <- inputGate(1,s_r[t],s_f[t])
            cur_wm <- ig$wm
            stripes_m <- ig$stripes_m
            f_stripes_m <- ig$f_stripes_m
        }

        ## We only need to select an action in this case
        for( t in 1:nqueries ) {
            ac <- selectAction(s_r[t])
            if( ac$action == s_f[t] )
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
