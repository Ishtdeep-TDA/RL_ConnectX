from Game import Game

class Arena():
    """
    Pits two agents against each other
    """

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        
    def play_game(self):
        
        '''
        Playing one game
        
        '''
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
            
            if opt_player == 1:
                action = self.player1.test_policy(game_copy,1)
            else:
                action = self.player2.test_policy(game_copy,-1)
            
            game.getNextState(action)
            r = game.getGameEnded(opt_player)
            
            #print the board
            p_print(game.env.board)
            print("result ", r)
            if r != 0:
                break
            opt_player = -opt_player
        