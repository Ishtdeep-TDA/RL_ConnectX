import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class my_mcts():
    '''
    Implementation of MCTS algorithm based on the lecture 9
    from David Silver's course on youtube.
    
    Before leaf node is found, we run UCB, and after that we run
    the current policy.
    
    Leaf node is defined as a node which has not been seen by the UCB algorithm,
    Each time a node is found which is not seen, it gets added into the UCB
    tree (which is S_counts).
    '''
    def __init__(self,cur_state,nnet,args):
        '''
        state - It is the root state (in this case the game (which is a wrapper over
        the simplifiedconnectx gym environment))
        cur_policy - This is the policy which is run when the leaf node is found
        args - this has arguments like exploration rate, num_mcts_sims
        
        '''
        self.args = args
        self.game = cur_state
        #This is the dictionary which keeps track of the score for all states and actions
        self.SA_score = {}
        #This stores the state action pair counts
        self.SA_counts = {}
        # This is the dictionary which counts the number of times a state is visited
        self.S_counts = {}
        # This stores the total number of times a simulation has been run
        self.total_counts = 0
        # This is the exploration constant, which governs how much we should explore
        # Higher c means higher exploration
        self.c = args["c"]
        # This could be a NN or any policy. The input would be the board
        # The output should be a probability distribution of moves
        self.nnet = nnet
        
    def getActionProb(self,game,opt_player):
        '''
        This function runs all the simulations of the MCTS starting
        from the root node and then returns the final prob distribution
        
        arguments - 
        game: this is the current state of the board
        opt_player: this is the player to optimize for
        '''
        for i in range(self.args["num_mcts_sims"]):
            print("inside mcts, iteration number",i)
            self.search(game.create_copy(),opt_player)
        
        s = game.stringRepresentation()
        # Lets get the scores of children of the current state s and exponentiate to
        # implement softmax
        prob = [math.exp(self.SA_score[(s,a)]) if (s,a) in self.SA_score else 0\
                for a in range(game.getActionSize())]
        print("exponentiated",prob)
        # Now we need to normalize these scores
        total = sum(prob)
        
        if total == 0:
            # should never reach here
            print("There divide by zero in getActionProb")
        # Normalizing the probabilities
        prob = [x/total for x in prob]
        
        
        # The below might be implemented in a different branch 
        # (If we use the below code) we would be implementing TD sort of algo
        # # The value of being the state s is 
        # # following the policy after all searches, what is the value
        # # Lets make this optimistic, which means that the current state is 
        # # as good as the next best state and so
        # index_max = max(range(len(prob)), key=prob.__getitem__)
        # v = self.SA_score[(s,index_max)] / self.SA_counts[(s,index_max)]
        # print("The value of being in state S: ", v)
        print("The probabilities for state S are :", prob)
        
        return prob
        
    def search(self,game,opt_player):
        '''
        This actually runs 1 episode from the total number of simulations
        
        If we are not at the leaf node, we should use UCB to pick which node to expand,
        If we are at the leaf node, we should use the current policy to expand the tree
        
        '''
        s = game.stringRepresentation()
        
        
        #check if the game has ended
        game_result = game.getGameEnded(opt_player)
        if game_result != 0:
            # The game has ended
            return game_result
        #check for leaf node
        if s not in self.S_counts: # If this is true then this is a leaf node
            # Now we will play a full game from here, using the NN and record the result
            canonicalBoard = game.getCanonicalForm()
            q_score,v_score = self.nnet.predict(canonicalBoard)
            valids = game.getValidMoves()
            q_score = q_score * valids  # masking invalid moves
            sum_Ps_s = np.sum(q_score)
            if sum_Ps_s > 0:
                # q_score represents the prob distribution of our NN
                q_score /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                q_score = q_score + valids
                q_score /= np.sum(q_score)
            
            # Lets choose a move from the prob distribution of this NN
            action = np.random.choice(game.getActionSize(),p = q_score)
            
            nest_s, reward, done, info = game.getNextState(action)
            result = self.search(game,opt_player)
            
            if result != 0:
                self.SA_score[(s, action)] = result
                self.SA_counts[(s, action)] = 1
                self.S_counts[s] = 1
                return result
            
        else: # not a leaf node
            # We have to apply UCB as this has been visited before
            best_action = None
            best_action_score = -float("inf")
            valids = game.getValidMoves()
            for action,move in enumerate(valids):
                # action is valid
                if move != 0:
                    #check if we have visited this action
                    if (s,action) in self.SA_score:
                        UCB_score = self.SA_score[(s,action)] + self.c*(math.sqrt(self.total_counts))/(self.SA_counts[(s,action)])
                    else:
                        # This is max because self.SA_counts[(s,action)] = 0 which is the denominator
                        UCB_score = float("inf")
                    if UCB_score > best_action_score:
                        best_action_score = UCB_score
                        best_action = action
            
            # Now that we have the best action, lets move it and continue
            nest_s, reward, done, info = game.getNextState(best_action)
            result = self.search(game,opt_player)
            self.S_counts[s] += 1
            # updating the score for the current state
            if (s, best_action) in self.SA_score:
                self.SA_score[(s, best_action)] = (self.SA_counts[(s, best_action)] * self.SA_score[(s, best_action)] + result) / (self.SA_counts[(s, best_action)] + 1)
                self.SA_counts[(s, best_action)] += 1
            else:
                self.SA_score[(s, best_action)] = result
                self.SA_counts[(s, best_action)] = 1
            #passing the result to the parent
            return result
