import numpy as np
import math
import time
from policy import random_policy
from Coach import p_print 
from collections import defaultdict

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
    def __init__(self,cur_state,args):
        '''
        state - It is the root state (in this case the game (which is a wrapper over
        the simplifiedconnectx gym environment))
        cur_policy - This is the policy which is run when the leaf node is found
        args - this has arguments like exploration rate, num_mcts_sims
        
        '''
        self.args = args
        self.game = cur_state
        #This is the dictionary which keeps track of the score for all states and actions
        self.SA_score = defaultdict(float)
        #This stores the state action pair counts
        self.SA_counts = defaultdict(int)
        # This is the dictionary which counts the number of times a state is visited
        self.S_counts = {}
        # This stores the total number of times a simulation has been run
        self.total_counts = 0
        # This is the exploration constant, which governs how much we should explore
        # Higher c means higher exploration
        self.c = args["c"]
        # This is used by the search method to keep track of the previous move played
        self.last_state_action = None # it is the (state,action) tuple for the previous move


    def getActionProb(self,game,opt_player):
        '''
        This function runs all the simulations of the MCTS starting
        from the root node and then returns the final prob distribution
        
        arguments - 
        game: this is the current state of the board
        opt_player: this is the player to optimize for
        '''
        start = time.time()
        for i in range(self.args["num_mcts_sims"]):
            self.search(game.create_copy(),opt_player)
        print(f"total time taken to run {self.args['num_mcts_sims']} iterations is {time.time() - start}")
        s = game.stringRepresentation()
        # Lets get the scores of children of the current state s and exponentiate to
        prob = [self.SA_score[(s,a)] if (s,a) in self.SA_score else 0\
                for a in range(game.getActionSize())]
        action = max(range(len(prob)), key=prob.__getitem__)
        prob = [0]*len(prob)
        prob[action] = 1

        print("The probability distribution is ", prob)
        
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

        if self.last_state_action not in self.SA_counts: # If this is true then this is a leaf node
            action = random_policy(game)
            nest_s, reward, done, info = game.getNextState(action)
            result = self.policy_search(game,opt_player)
            
            if result != 0:
                self.SA_score[(s, action)] = result
                self.SA_counts[(s, action)] = 1
                self.S_counts[s] = 1
                self.last_state_action = (s,action)
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
            self.S_counts[s] += 1
            self.SA_counts[(s, best_action)] += 1
            self.last_state_action = (s,best_action)
            
            result = self.search(game,opt_player)
            # updating the score for the current state
            self.SA_score[(s, best_action)] += result
            #passing the result to the parent
            return result

    def policy_search(self,game,opt_player):
        '''
        This search is run when the leaf node is reached
        '''
        action = random_policy(game)
        nest_s, reward, done, info = game.getNextState(action)
        result = self.policy_search(game,opt_player)
        
        return result
