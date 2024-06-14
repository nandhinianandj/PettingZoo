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
        self.squares = list(range(width)) * breadth
        self.resources =  np.array(np.random.randn(width, breadth))
        plt = sns.heatmap(self.resources, annot=False)
        plt.set(xticklabels=[])
        plt.set(yticklabels=[])
        #plt.tick_
        fig = plt.get_figure()
        fig.savefig("img/ecosystem.png")
        fig.savefig("img/ecosystem_initial.png")

    def update_resources(self, rewards):
        for player, reward in rewards.items():
            if isinstance(reward, dict):
                self.resources += reward['global']
                self.resources[self.resources < 0] = 0
                print(self.resources)
                plt = sns.heatmap(self.resources, annot=False)
                plt.set(xticklabels=[])
                plt.set(yticklabels=[])
                fig = plt.get_figure()
                fig.savefig("img/ecosystem.png")
            else:
                continue
        pass

    def check_game_over(self):
        #winner = self.check_for_winner()

        if winner == -1 and all(square in [1, 2] for square in self.squares):
            # tie
            return True
        elif winner in [1, 2]:
            return True
        else:
            return False

    def __str__(self):
        return str(self.squares)
