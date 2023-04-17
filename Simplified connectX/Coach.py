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

    def __init__(self, game, nnet, args):
        self.game = game
#         self.nnet = nnet
        self.nnet = nnet  # only 1 network in alphazero
        self.args = args
        self.mcts = my_mcts(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

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
        trainExamples = []
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
            canonicalBoard = game.getCanonicalForm()
            # temp = int(episodeStep < self.args.tempThreshold)
            
            game_copy = game.create_copy()
            self.mcts = my_mcts(game_copy, self.nnet, self.args)
            pi = self.mcts.getActionProb(game_copy, opt_player)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, opt_player, p, None])
            
            action = np.random.choice(len(pi), p=pi)
            nest_s, reward, done, info = game.getNextState(action)
            r = game.getGameEnded(opt_player)
            
            #print the board
            p_print(game.env.board)
            print("result ", r)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != opt_player))) for x in trainExamples]
            opt_player = -opt_player

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1, self.args["num_iters"] + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args["maxlenOfQueue"])

                for _ in tqdm(range(self.args["numEps"]), desc="Self Play"):
                    self.mcts = my_mcts(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args["numItersForTrainExamplesHistory"]:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args["checkpoints"], filename='temp.pth.tar')

            self.nnet.train(trainExamples)

        
    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args["checkpoint"]
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args["load_folder_file"][0], self.args["load_folder_file"][1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
            
    
        
        