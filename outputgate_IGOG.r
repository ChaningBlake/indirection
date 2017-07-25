#############################################################################
##
## Indirection - phase 2: output gating 
## Role/filler recall task
## 2017 Mike Jovanovich
##
## In this version OG sees WM after it has been update by IG within a timestep
##
#############################################################################

## Usage
# Rscript outputgate.r 1 100000 3 10 3 C C D F

#############################################################################
## Parameter initialization
#############################################################################

source('hrr.r')
source('sentenceSets.r')
require(methods)
require(Matrix)

## Build args dictionary
argnames <- list('seed','ntasks','nstripes','nfillers','nroles','state_cd','sid_cd','interstripe_cd',
  'use_sids_input','use_sids_output')
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
gono_m <- replicate(2,hrr(n,normalized=TRUE))
gono_o <- replicate(2,hrr(n,normalized=TRUE))

## TD weight vectors
## A suffix of '_m' inicates relation to the maintenance layer NN
## To index: [,stripe]
W_g <- replicate(nstripes,hrr(n,normalized=TRUE))

## Action weight vectors
## There are 'nfillers' output units for this network
## Each output unit should come to represent the desired filler
## To index: [,output_unit]
W_a <- replicate(nfillers,hrr(n,normalized=TRUE))

## TD parameters
default_reward <- 0.0
success_reward <- 1.0
bias_g <- 1.0
bias_a <- 1.0
gamma_g <- 1.0
gamma_a <- 1.0
lambda_g <- 0.9
lrate_g <- 0.1
lrate_a <- 0.1
#ac_softmax_t <- .025    ## Higher temperatures will increase chance of random action
ac_softmax_t <- .125    ## Higher temperatures will increase chance of random action
epsilon_g <- .025

## Task parameters
max_tasks <- as.integer(args[which(argnames=='ntasks')])
cur_task <- 1
cur_block_task <- 1
block_tasks_correct <- 0

## Get training and test sets
train_set_size <- 200
test_set_size <- 100

## Choose testing protocol
sets <- standardGeneralization(nroles,nfillers,train_set_size,test_set_size)

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
## selectAction:
##
## This function handles action selection. The hidden layer takes as input
## working memory contents along with the role that is being queried.
## The max output unit is used to determine trial correctness.
##
#############################################################################
selectAction <- function() {

    ## Get NN output and output unit (action) with max value
    output <- (apply(W_a,2,nndot,cur_wm_o) + bias_a)
    action <- softmax_select(output)
    
    return(list(
        val = output,
        action = action
    ))
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
    state_wm <- convolve(state,cur_wm_m)

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

        ## Epsilon soft policy
        r <- runif(1)
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
    state_wm <- convolve(state,cur_wm_m)

    ## This is state_wm convolved with both the go and no go hrrs
    state_wm_gono <- apply(gono,2,convolve,state_wm)

    ## Build to matrix of eligility trace vectors that will be returned
    ## for TD updates; default to no go
    elig <- replicate(nstripes,state_wm_gono[,2])
    vals <- rep(0.0,nstripes)
    open <- rep(FALSE,nstripes)

    ## Determine if open or close value is better
    for( i in 1:nstripes ) {

        temp_vals <- (apply(state_wm_gono,2,nndot,W_o[,i]) + bias_o)
        vals[i] <- max(temp_vals)

        ## Epsilon soft policy
        r <- runif(1)
        if( r < epsilon_o ) {
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
            stripes_o[,i] <- stripes_mo[,i]
            f_stripes_o[i] <- f_stripes_m[i]
        }
    }

    ## Get updated wm representation
    if( interstripe_cd == 'D' ) {
        wm <- cnorm(apply(stripes_o,1,sum))
        if( is.nan(wm[1]) )
            wm <- hrr_o ## TODO: hrr_z?
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

#############################################################################
## inputOutputGate:
##
## This function handles gating for a single timestep/operation in the task.
## At each timestep a stripe can either retain its contents or update with the 
## provided filler. 
##
## Multiple stripes update in a single timestep.
##
#############################################################################
inputOutputGate <- function(state,f=-1) {

    ## Start with a blank slate
    stripes_o <- replicate(nstripes,hrr_i)
    f_stripes_o <- replicate(nstripes,'I')

    ## Get NN output unit values
    ## Note that state_wm is for the previous timestep
    ## We will return this to use in the eligibility trace
    state_wm <- convolve(state,cur_wm_m)

    ## Build to matrix of eligility trace vectors that will be returned
    ## for TD updates; default to no go
    elig <- replicate(nstripes,hrr_z)
    vals <- rep(-999.9,nstripes)
    open_m <- rep(FALSE,nstripes)
    open_o <- rep(FALSE,nstripes)

    ## Loop through each stripe; each gate (input or output) can
    ## either be open or closed (1 or 2)
    for( s in 1:nstripes) {

        ## Epsilon soft policy
        r <- runif(1)
        if( r < epsilon_g ) {
            m <- 2
            o <- 2

            ## Pick a random open or close state for m
            ## Keep IG closed on no fill trials to keep a level playing field with other model
            if( f != -1 ) {
                r <- runif(1)
                if( r > .5 ) {
                    m <- 1
                }
            }
            ## Pick a random open or close state for o
            r <- runif(1)
            if( r > .5 ) {
                m <- 1
            }

            state_wm_gono <- convolve(convolve(state_wm,gono_m[,m]),gono_o[,o])
            temp_val <- nndot(W_g[,s],state_wm_gono) + bias_g
            vals[s] <- temp_val
            elig[,s] <- state_wm_gono
            open_m[s] <- m == 1
            open_o[s] <- o == 1

        } else {
            for( m in 1:2 ) {
                ## Keep IG closed on no fill trials to keep a level playing field with other model
                if( f == -1 && m == 1 )
                    next
                for( o in 1:2 ) {
                    state_wm_gono <- convolve(convolve(state_wm,gono_m[,m]),gono_o[,o])
                    temp_val <- nndot(W_g[,s],state_wm_gono) + bias_g

                    if( temp_val > vals[s] ) {
                        vals[s] <- temp_val
                        elig[,s] <- state_wm_gono
                        open_m[s] <- m == 1
                        open_o[s] <- o == 1
                    }
                }
            }
        }
    }

    ## Update WM_m contents 
    ## We are convolving the fill with the appropriate SID
    for( i in 1:nstripes ) {
        if( open_m[i] ) {
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
        wm_m <- cnorm(apply(stripes_m,1,sum))
        if( is.nan(wm_m[1]) )
            wm_m <- hrr_o
    } else {
        wm_m <- mconvolve(stripes_m)
    }

    ## Update WM_o contents 
    ## We are convolving the fill with the appropriate SID
    for( i in 1:nstripes ) {
        if( open_o[i] ) {
            stripes_o[,i] <- stripes_mo[,i]
            f_stripes_o[i] <- f_stripes_m[i]
        }
    }

    ## Get updated wm representation
    if( interstripe_cd == 'D' ) {
        wm_o <- cnorm(apply(stripes_o,1,sum))
        if( is.nan(wm_o[1]) )
            wm_o <- hrr_o ## TODO: hrr_z?
    } else {
        wm_o <- mconvolve(stripes_o)
    }

    return(list(
        wm_m = wm_m,
        wm_o = wm_o,
        elig = elig,
        vals = vals,
        stripes_m = stripes_m,
        stripes_mo = stripes_mo,
        f_stripes_m = f_stripes_m,
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
    elig <- replicate(nstripes,rep(0,n)) # There is an elig. trace for each BG input gate
    prev_val <- rep(0.0,nstripes)     # prev. timestep values for maintenance NN output layer units
    cur_wm_m <- hrr_o
    cur_wm_o <- hrr_o ## TODO: hrr_z?

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
        ## Input / Output gating
        #############################################################################

        ## Update WM input layer global variables
        g <- inputOutputGate(state,s_f[p[t]])
        cur_wm_m <- g$wm_m
        stripes_m <- g$stripes_m
        stripes_mo <- g$stripes_mo
        f_stripes_m <- g$f_stripes_m
        cur_wm_o <- g$wm_o
        f_stripes_o <- g$f_stripes_o

        #############################################################################
        ## Action selection
        #############################################################################
        
        ac <- selectAction()

        #############################################################################
        ## Neural network and TD training for this trial
        #############################################################################

        ## INPUT OUTPUT GATE
        error <- (reward + gamma_g * g$vals) - prev_val
        for( i in 1:nstripes ) {
            W_g[,i] <- W_g[,i] + lrate_g * error[i] * elig[,i]
            elig[,i] <- cnorm(lambda_g * elig[,i] + g$elig[,i])
        }
        prev_val <- g$vals 

        ## ACTION SELECTION
        ## This is not TD learning - just a NN
        answer <- rep(0,nfillers)
        answer[s_f[p[t]]] <- 1
        error <- answer - ac$val
        W_a <- (W_a + lrate_a * t(error %*% t(cur_wm_o)))

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

        ## TODO: no IG here

        #############################################################################
        ## Input / Output gating
        #############################################################################

        ## Update WM input layer global variables
        g <- inputOutputGate(state,-1)
        cur_wm_m <- g$wm_m
        stripes_m <- g$stripes_m
        stripes_mo <- g$stripes_mo
        f_stripes_m <- g$f_stripes_m
        cur_wm_o <- g$wm_o
        f_stripes_o <- g$f_stripes_o

        #############################################################################
        ## Action selection
        #############################################################################
        
        ac <- selectAction()

        #############################################################################
        ## Neural network and TD training for this trial
        #############################################################################

        ## INPUT OUTPUT GATE
        error <- (reward + gamma_g * g$vals) - prev_val
        for( i in 1:nstripes ) {
            W_g[,i] <- W_g[,i] + lrate_g * error[i] * elig[,i]
            elig[,i] <- cnorm(lambda_g * elig[,i] + g$elig[,i])
        }
        prev_val <- g$vals 

        ## ACTION SELECTION
        ## This is not TD learning - just a NN
        answer <- rep(0,nfillers)
        answer[s_f[p[t]]] <- 1
        error <- answer - ac$val 
        W_a <- (W_a + lrate_a * t(error %*% t(cur_wm_o)))

        ## Determine correctness
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

            ## INPUT OUTPUT GATE
            error <- reward - g$vals 
            for( i in 1:nstripes ) {
                W_g[,i] <- W_g[,i] + lrate_g * error[i] * elig[,i]
            }
        }
    }

    #############################################################################
    ## Task wrapup
    #############################################################################

    ## Debug prints
    #if( FALSE ) {
    #if( TRUE ) {
    #if( cur_task %% 100 == 0 ) {
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
        cat(round(softmax(ac$val),4))
        cat('\n')
        cat(sprintf('Requested Role: %d\n',p[t]))
        cat(sprintf('Correct Action: %d\n',s_f[p[t]]))
        cat('\n')
    }

    ## Increment task tally
    cur_task <- cur_task + 1
}

## Print final results
#cat(sprintf('%d\n',cur_task))
cat(sprintf('Final Block Accuracy: %.2f\n',(block_tasks_correct/200)*100))

#############################################################################
## Generalization Test
#############################################################################
if(FALSE) {
    novel_tasks_correct <- 0
    for( i in 1:test_set_size ) {
        correct_trial <- rep(0,nroles)

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
            ## Input Gate
            ig <- inputGate(1,p[t],s_f[p[t]])
            cur_wm_m <- ig$wm
            stripes_m <- ig$stripes_m
            stripes_mo <- ig$stripes_mo
            f_stripes_m <- ig$f_stripes_m
        }

        ## We only need to select an action in this case
        for( req in 1:nroles ) {
            ac <- selectAction()
            if( ac$action == s_f[req] ) {
                correct_trial[req] <- 1
            }
        }

        ## Determine if entire sequence is correct
        ## Not sure if we'll want to modify protocol above to train for this
        if( sum(correct_trial) == nroles )
            novel_tasks_correct <- novel_tasks_correct + 1
    }

    ## Print final results
    cat(sprintf('Generalization Accuracy: %d\n',novel_tasks_correct))
} ## End generalization test
