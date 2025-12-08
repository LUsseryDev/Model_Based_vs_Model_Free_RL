import EnvUtil
from MiscUtil import *
import keyboard


class RandomAction:
    def __init__(self, game:str, episodes:int, max_steps:int, stack_n_frames:int, n_action_repeats:int, print_freq:int):
        self.game = game
        self.env = EnvUtil.getEnvList([game])[0]
        self.episodes = episodes
        self.stack_n_frames = stack_n_frames
        self.n_action_repeats = n_action_repeats
        self.max_steps = max_steps
        self.print_freq = print_freq
        self.paused = False

    def __str__(self):
        return 'RandomAction'

    def pause(self, event):
        self.paused = True

    def run(self):

        keyboard.on_press_key('p', self.pause)

        rewards = []
        for i in range(self.episodes): #loop per episode
            #episode setup
            e_reward = 0
            frame, info = self.env.reset()
            preprocessor = Preprocessor(frame, self.stack_n_frames)
            obs = preprocessor.get_obs(frame)
            lives = info['lives']


            for s in range(self.max_steps): #loop per step
                #get action
                #feed obs to model
                action = self.env.action_space.sample()

                #run action
                for j in range(self.n_action_repeats - 1):
                    self.env.step(action)
                next_frame, reward, terminated, truncated, info = self.env.step(action)


                e_reward += clip_reward(reward)
                obs = preprocessor.get_obs(next_frame)
                if lives != info['lives'] or terminated or truncated:
                    break

            rewards.append(e_reward)

            if i % self.print_freq == 0:
                print("Episode {}| Avg. Reward {:.3f}".format(i, np.mean(rewards[-self.print_freq:])))

            if self.paused:
                print("\nPaused")
                print("input \"1\" to quit, input anything else to continue (that doesn't start with \'p\')")
                choice = input("Enter your choice: ")

                #this is stupid, thank you input buffer
                choice = choice.lstrip("p")

                if choice == "1":
                    self.save()
                    return
                print("resuming")
                self.paused = False

    def train(self):
        self.run()

    def save(self):
        return

    def load(self):
        return