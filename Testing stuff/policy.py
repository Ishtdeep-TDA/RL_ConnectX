import numpy as np
import math
import random
import os
import time

class RandomPolicy():
    """
    This is an agent which plays randomly.
    """

    def __init__(self):
        pass
    
    def train_policy(self, game):
        '''
        This is used during the training process (training policy)
        
        '''
        possible_moves = game.env.get_possible_actions()
        action = random.choice(possible_moves)
        return action
    
    
    def test_policy(self, game,opt_player):
        '''
        This is called during the testing process could be the same 
        as the training process and could be different 
        (testing policy)
        
        game - current state of the game
        opt_player - the player to optimize for (the current
        player's turn)
        
        '''
        possible_moves = game.env.get_possible_actions()
        action = random.choice(possible_moves)
        return action
    
