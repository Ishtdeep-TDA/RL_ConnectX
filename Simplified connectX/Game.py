
from tqdm import tqdm
from collections import deque
import random
import numpy as np
from simplified_connectx import *


class Game():
    """
    This is a wrapper over the gym environment and has some extra functions implemented
    within it over the ones that are in the gym environment.

    Use 1 for player1 and -1 for player2.


    if config = None then default config (6,7) board
    """
    def __init__(self,config = None):
        self.env = SimplifiedConnectX()
        self.config = config
        
        
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
                        
        """
        self.env.reset()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.env.n_rows,self.env.n_cols)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.env.n_cols
    
    def getNextState(self,action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player 
            (it is the index of the action that is
            chosen)

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # Since we have the board, player and the action lets reset the environment?
        next_board, reward, done, info  = self.env.step(action)
        
#         if reward == -1 and player == -1:
#             reward = 1
#         elif reward == -1 and player == 1:
#             reward = -1
#         elif reward == 1 and player == -1:
#             reward = -1
#         elif reward == 1 and player == 1:
#             reward = 1
            
        return next_board, reward ,done, info
        
        
    def getValidMoves(self):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        valid_moves = [0]*self.env.n_cols
        moves = self.env.get_possible_actions()
        for i in moves:
            valid_moves[i] = 1
        
        return valid_moves
          
    def is_win(columns,rows,inarow,board, column, mark, has_played=True):
        '''
        Gets us if a player has won or not

        inputs:

        rows - number of rows
        columns - number of columns
        inarow - How many in a row required to win.
        mark - the current player (2 for p2 , 1 for p1)
        column - the current player's move

        '''
        columns = columns
        rows = rows
        inarow = inarow - 1
        row = (
            min([r for r in range(rows) if board[column + (r * columns)] == mark])
            if has_played
            else max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
        )
        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                    r < 0
                    or r >= rows
                    or c < 0
                    or c >= columns
                    or board[c + (r * columns)] != mark
                ):
                    return i - 1
            return inarow

        return (
            count(1, 0) >= inarow  # vertical.
            or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
            or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
            or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
        )

    def getGameEnded(self, optim_player):
        """
        Input:
            board: current board
            optim_player: player that we are optimizing for

        Returns:
            r: 0 if game has not ended. 1 if optim_player won, -1 if optim_player lost,
               -0.5 for a draw
               
        """
        temp_rew = self.env.get_reward()
        # since the current get_reward optimizes for only player 1
        if optim_player == 1:
            return temp_rew
        # when we are optimizing for player 2
        else:
            if temp_rew == 1:
                return -1
            if temp_rew == -1:
                return 1
            else:
                return temp_rew
            
    def getCanonicalForm(self):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
            
        
        if player = 1
        1st layer for player 1 coins
        2nd layer for player 2 coins
        3rd layer for empty slots
        
        if player = 2
        1st layer for player 2 coins
        2nd layer for player 1 coins
        3rd layer for empty slots
        
        """
        
        temp = np.array(self.env.board)
        temp = temp.reshape(6,7)
        temp = np.expand_dims(temp,axis = 0)
        

        #indicates player 1's pieces
        p1 = np.copy(temp)
        p1[p1 != 1] = 0

        #indicates player 2's pieces
        p2 = np.copy(temp)
        p2[p2 != 2] = 0
        p2[p2 == 2] = 1

        # Indicates the empty spaces
        temp[temp == 0] = 3
        temp[temp != 3] = 0
        temp[temp == 3] = 1
        
        # NOTE this returns a channel first version.
        if self.env.current_player == 1:
            output = np.concatenate([p1,p2,temp],axis = 0)
            return output
        else:
            output = np.concatenate([p2,p1,temp],axis = 0)
            return output
    
    def invert_canonical(self,canonical_board,player):
        '''
        From the canonical board, get the original game board that
        the environment wants
        
        player - player with which this board was made.
        '''
        if player == 1:
            output = canonical_board[1]*2 + canonical_board[0]
            output = output.flatten()
            
        return output2
    
    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board (canonical form)
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # lateral symmetry does exist
        
        
        return [(board,pi)]

    def stringRepresentation(self):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return "".join(str(x) for x in self.env.board)
    def create_copy(self):
        '''
        Create a copy of the current game object.
        '''
        copy_object = Game()
        copy_object.env = self.env.create_copy()
        return copy_object
    def get_possible_actions(self,):
        return self.env.get_possible_actions()