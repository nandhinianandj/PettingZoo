import numpy as np
import seaborn as sns

class Board:
    def __init__(self, width=8, breadth=8):
        # internally self.board.squares holds a flat representation of tic tac toe board
        # where an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # where indexes are column wise order
        # 0 3 6
        # 1 4 7
        # 2 5 8

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        rng = np.random.default_rng(12345)
        self.squares = list(range(width)) * breadth
        self.resources =  np.array(rng.integers(low=0, high=2**8, size=64)).reshape(width, breadth)
        plt = sns.heatmap(self.resources, annot=True)
        plt.set(xticklabels=[])
        plt.set(yticklabels=[])
        #plt.tick_
        fig = plt.get_figure()
        fig.savefig("img/ecosystem.png")
        fig.savefig("img/ecosystem_initial.png")

    #def setup(self):
    #    pass
    #    self.calculate_winners()

    def update_resources(self, rewards):
        if 'global' in rewards:
            self.resources += rewards['global']
            print(self.resources)
            plt = sns.heatmap(self.resources, annot=False)
            plt.set(xticklabels=[])
            plt.set(yticklabels=[])
            #plt.tick_
            fig = plt.get_figure()
            fig.savefig("img/ecosystem.png")
        pass

    def check_for_winner(self):
        winner = -1
        for combination in self.winning_combinations:
            states = []
            for index in combination:
                states.append(self.squares[index])
            if all(x == 1 for x in states):
                winner = 1
            if all(x == 2 for x in states):
                winner = 2
        return winner

    def check_game_ove(self):
        winner = self.check_for_winner()

        if winner == -1 and all(square in [1, 2] for square in self.squares):
            # tie
            return True
        elif winner in [1, 2]:
            return True
        else:
            return False

    def __str__(self):
        return str(self.squares)
