import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
# from MCTS import MCTS
from my_mcts import my_mcts
from Game import Game
from pprint import pprint

log = logging.getLogger(__name__)

def p_print(board):
    temp = board
    temp = np.array(temp)
    temp = temp.reshape(6,7)
    pprint(temp)

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.mcts = my_mcts(self.game, self.args)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
#       reset the environment
        game = Game()
        game.getInitBoard()
        # This is the player that we are currently optimizing for
        # Changes when the turn changes
        opt_player = 1 # can be 1 or -1
        episodeStep = 0
        
        while True: 
            episodeStep += 1
            print("Currently on move ",episodeStep)

            game_copy = game.create_copy()
            self.mcts = my_mcts(game_copy, self.args)
            pi = self.mcts.getActionProb(game_copy, opt_player)
            
            action = np.random.choice(len(pi), p=pi)
            nest_s, reward, done, info = game.getNextState(action)
            r = game.getGameEnded(opt_player)
            
            #print the board
            p_print(game.env.board)
            print("result ", r)
            if r != 0:
                print("Result : ",r, "Optimizing player : ", opt_player)
                break

            opt_player = -opt_player


    
        
        