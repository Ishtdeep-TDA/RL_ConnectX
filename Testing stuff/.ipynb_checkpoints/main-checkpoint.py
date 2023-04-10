import logging

import coloredlogs
import time

from policy import RandomPolicy
from MCTS_Agent import MCTS_Agent
from Game import Game
from Arena import Arena

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

#NOTE - Total games played = numIters * numEps
# Total number of times NN is trained - NumIters
agent_1_args = {

    "numMCTSSims" : 100,
    

}


def main():
    print("Testing the random agent")
    
    agent1 = MCTS_Agent(Game(),RandomPolicy(),agent_1_args)
    agent2 = RandomPolicy()
    arena = Arena(agent1,agent2)
    
    start = time.time()
    arena.play_game()
    end = time.time()
    
    print("The total time taken to play 1 game is ", end-start)
    
    
    


if __name__ == "__main__":
    main()
