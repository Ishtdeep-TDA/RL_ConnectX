import gym

class SimplifiedConnectX(gym.Env):
#     metadata = {"render_modes":["None"]} idk if I should include or no
    def __init__(self,render_mode = None, board_size = (6,7)):
        '''
        board_size = tuple of 2 (rows,columns)
        
        '''
        # number of rows
        self.n_rows = board_size[0]
        # number of columns
        self.n_cols = board_size[1]
        # board represented by a list
        # player 1's tokens are 1, player -1 tokens are 2
        self.board = [0] * (self.n_rows*self.n_cols)
        # This stores the index of the top element of the board 
        # which is empty (so if n_row == self.top[i] then ith column
        # is full)
        self.top = [0]*self.n_cols
        # player can be 1 or -1 
        self.current_player = 1
        # If this is 1 then that means the game is over
        self.done = 0
        # Additional info if required
        self.info = {
            # represents the number of times step is called
            "n_turns":0,
        }
        
    def reset(self):
        self.board = [0] * (self.n_rows*self.n_cols)
        self.top = [0]*self.n_cols
        self.no_turns = 0
        self.current_player = 1
        self.done = 0
        self.info["n_turns"] = 0
        
        return self.board, self.info
    
    def step(self,action):
        '''
        action - integer which tells which column to put the coin in
        '''
        # check if it is possible to play the action
        if self.top[action] >= self.n_rows:
            raise Exception(f"Column {action} is full !!")
        # should never trigger this
        if self.done == 1:
            raise Exception("The game is already over !")
        token = 0
        if self.current_player == 1:
            token = 1
        else:
            token = 2
        index = (self.n_rows*self.n_cols) - self.top[action]*self.n_cols - (self.n_cols - action)
        self.board[index] = token 
        self.top[action] += 1
        self.current_player = -self.current_player
        
        reward = self.get_reward()
        
        if reward != 0:
            self.done = 1
        self.info["n_turns"] += 1
        
        return self.board, reward, self.done, self.info
        
    def get_reward(self):
        '''
        
        The reward is 1 if player 1 wins -1 if player
        -1 wins, -0.5 if a draw occurs, 0 if the game is not terminated.
        
        '''
        # testing for player 1 
        for i in range(self.n_cols):
            try:
                temp = self.is_win(self.n_cols,self.n_rows,4,self.board,i,1)
                # player 1 wins
                if temp == True:
                    return 1
            except:
                pass
        # testing for player 2 
        for i in range(self.n_cols):
            try:
                temp = self.is_win(self.n_cols,self.n_rows,4,self.board,i,2)
                # player 2 wins
                if temp == True:
                    return -1
            except:
                pass
            
        # check for a draw
        draw_flag = 1
        for i in self.top:
            if i < self.n_rows:
                draw_flag = 0
                break

        if draw_flag == 1:
            return -0.5
        
        return 0
    def get_possible_actions(self):
        '''
        returns an array with integers indicating that those
        columns are empty
        '''
        output = []
        for i in range(self.n_cols):
            if self.top[i] < self.n_rows:
                output.append(i)
        return output
        
        
    def is_win(self,columns,rows,inarow,board, column, mark, has_played=True):
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
    def create_copy(self):
        '''
        Takes the current environment and creates a copy of it
        
        '''
        new_env = SimplifiedConnectX(board_size = (self.n_rows,self.n_cols))
        new_env.board = self.board[:]
        new_env.top = self.top[:]
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.info = self.info
        
        return new_env
        
        
        
        
        
        