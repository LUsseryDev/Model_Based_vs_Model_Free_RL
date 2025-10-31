import gymnasium

from Algorithms import IMPALA
from Algorithms.IMPALA import Impala
from Algorithms.RandomAction import RandomAction
from EnvUtil import *
from MiscUtil import Preprocessor

if __name__ == '__main__':
    #global hyperparameters
    episodes = 10
    max_steps = 1000
    n_action_repeats = 4
    stack_n_frames = 4
    obs_size = (106, 140)
    print_freq = 100

    algorithm_list = [IMPALA, RandomAction]

    while True:
        print('1. train all algorithms')
        print('2. train single algorithm')
        print('3. run all algorithms')
        print('4. run single algorithm')
        print('5. create graph')
        print('0. exit')

        choice = input('Enter your choice: ')
        match choice:
            case '1':
                gameList = chooseGames()

                print(gameList)
                for game in gameList:
                    for alg in algorithm_list:
                        a = alg(game, episodes, max_steps, stack_n_frames, n_action_repeats, print_freq)
                        print("starting {}".format(a.__str__()))
                        a.train()
                        a.save()
                        print("{} done".format(a.__str__()))

                    impala = Impala(game, episodes, max_steps, stack_n_frames, n_action_repeats, print_freq)
                    impala.train()


                    #preprocessor = Preprocessor(env.reset()[0], stack_n_frames)
                    break


            case '2':
                print('2. train single algorithm')
            case '3':
                print('3. run all algorithms')
            case '4':
                print('4. run single algorithm')
            case '5':
                print('5. create graph')
            case '0':
                print('Shutting down')
                break
            case _:
                print('Invalid choice, please try again')
