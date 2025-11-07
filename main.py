import sys

import gymnasium

from Algorithms import IMPALA
from Algorithms.IMPALA import Impala
from Algorithms.RandomAction import RandomAction
from EnvUtil import *
from MiscUtil import Preprocessor

if __name__ == '__main__':
    #global hyperparameters
    episodes = 100
    max_steps = 1000
    n_action_repeats = 4
    stack_n_frames = 4
    obs_size = (106, 140)
    print_freq = 10

    algorithm_list = ["Impala", "RandomAction"]
    algorithm_objects = []

    #if there are command line arguments, use the arguments instead of interaction with user
    if len(sys.argv) > 1:
        #to implement
        pass

    else:
        while True:
            print('1. train all algorithms')
            print('2. train single algorithm')
            print('3. run all algorithms')
            print('4. run single algorithm')
            print('0. exit')

            choice = input('Enter your choice: ')
            match choice:
                case '1':
                    gameList = chooseGames()

                    #print(gameList)
                    for game in gameList:
                        print("starting game {}".format(game))
                        for alg in algorithm_list:
                            a = cls = globals()[alg](game, episodes, max_steps, stack_n_frames, n_action_repeats, print_freq)
                            print("starting {}".format(a.__str__()))
                            a.train()
                            a.save()
                            algorithm_objects.append(a)
                            print("{} done".format(a.__str__()))

                        # rand = RandomAction("Asteroids", episodes, max_steps, stack_n_frames, n_action_repeats, print_freq)
                        # rand.train()
                        # break

                        # impala = Impala("Asteroids", episodes, max_steps, stack_n_frames, n_action_repeats, print_freq)
                        # impala.train()
                        # algorithm_objects.append(impala)
                        # break

                case '2':
                    print('2. train single algorithm')
                case '3':
                    print('3. run all algorithms')
                case '4':
                    print('4. run single algorithm')
                case '0':
                    print('Shutting down')
                    break
                case _:
                    print('Invalid choice, please try again')
