import sys
import time
from Algorithms.IMPALA import Impala
from Algorithms.RandomAction import RandomAction
from EnvUtil import *


if __name__ == '__main__':
    #global hyperparameters
    episodes = 50
    max_steps = 1000
    n_action_repeats = 4
    stack_n_frames = 4
    obs_size = (106, 140)
    print_freq = 10

    algorithm_list = ["Impala", "RandomAction"]
    #algorithm_list = ["Impala"]
    algorithm_objects = []

    #if there are command line arguments, use the arguments instead of interaction with user
    if len(sys.argv) > 1:
        #to implement
        pass

    else:
        while True:
            print('1. train all algorithms')
            print('2. train single algorithm')
            print('0. exit')

            choice = input('Enter your choice: ')
            match choice:
                case '1':
                    gameList = chooseGames()
                    #gameList = ["Asteroids", "Boxing", "Krull", "Pong"]

                    #print(gameList)
                    for game in gameList:
                        print("starting game {}".format(game))
                        for alg in algorithm_list:
                            a = globals()[alg](game, episodes, max_steps, stack_n_frames, n_action_repeats, print_freq)
                            print("starting {}".format(a.__str__()))
                            start = time.perf_counter()
                            a.train()
                            end = time.perf_counter()
                            a.save()
                            algorithm_objects.append(a)
                            print("{} done, training time was {} seconds".format(a.__str__(), str(end - start)))


                case '2':
                    print("Available Algorithms:")
                    print(algorithm_list)
                    while True:
                        alg = input("Enter algorithm: ")
                        if alg in algorithm_list:
                            print("selected algorithm: {}".format(alg))
                            break
                        else:
                            print("Invalid algorithm, please try again")
                    gameList = chooseGames()
                    for game in gameList:
                        print("starting game {}".format(game))
                        a = globals()[alg](game, episodes, max_steps, stack_n_frames, n_action_repeats, print_freq)
                        print("starting {}".format(a.__str__()))
                        start = time.perf_counter()
                        a.train()
                        end = time.perf_counter()
                        a.save()
                        algorithm_objects.append(a)
                        print("{} done, training time was {} seconds".format(a.__str__(), str(end - start)))

                case '0':
                    print('Shutting down')
                    break
                case _:
                    print('Invalid choice, please try again')
