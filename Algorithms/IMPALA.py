import time

import keyboard
import torch
import gymnasium as gym

import EnvUtil
from MiscUtil import *
import torch.multiprocessing as mp


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, action_space):
        super(NeuralNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            torch.nn.Linear(in_features=5632, out_features=256), #going to be wrong input features
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=action_space)
        )

    def forward(self, x):
        return self.network(x)

def actor(game:str, tqueue: mp.Queue, mqueue: mp.Queue, is_done:mp.Event(), is_paused:mp.Event(), trajectory_length:int, stack_n_frames:int, n_action_repeats:int):
    process = mp.current_process()
    print("starting actor with pid "+str(process.pid))
    env = EnvUtil.getEnvList([game])[0]
    network = NeuralNet(stack_n_frames, env.action_space.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame, info = env.reset()
    preprocessor = Preprocessor(frame, stack_n_frames)
    obs = preprocessor.get_obs(frame)
    lives = info['lives']

    #main loop
    while True:
        trajectory = []

        #update network
        if not mqueue.empty():
            network.load_state_dict(mqueue.get())

        #make a trajectory
        for i in range(trajectory_length):
            # get action
            with torch.no_grad():
                action_probs = network.forward(torch.from_numpy(obs).float().unsqueeze(0)).detach()
                action = torch.multinomial(torch.nn.functional.softmax(action_probs, dim=1), 1)

            # run action and save to trajectory
            for j in range(n_action_repeats - 1):
                env.step(action)
            next_frame, reward, terminated, truncated, info = env.step(action)
            trajectory.append((obs, action_probs, clip_reward(reward)))

            #ready for next step
            obs = preprocessor.get_obs(next_frame)
            if lives != info['lives'] or terminated or truncated:
                frame, info = env.reset()
                preprocessor = Preprocessor(frame, stack_n_frames)
                obs = preprocessor.get_obs(frame)
                lives = info['lives']

        tqueue.put(trajectory)


        if is_done.is_set():
            tqueue.cancel_join_thread()
            env.close()
            return


        while is_paused.is_set():
            time.sleep(2)
            if is_done.is_set():
                tqueue.cancel_join_thread()
                env.close()
                return

def learner(game:str, tqueue: mp.Queue, mqueues: list[mp.Queue], is_done:mp.Event(), is_paused:mp.Event(), trajectory_length:int, stack_n_frames:int, n_action_repeats:int, episodes:int, print_freq:int):
    process = mp.current_process()
    print("starting learner with pid "+str(process.pid))
    env = EnvUtil.getEnvList([game])[0]
    network = NeuralNet(stack_n_frames, env.action_space.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rewards = []
    for i in range(episodes):

        #do useful things that learn
        print("trajectory queue size {}".format(tqueue.qsize()))
        time.sleep(1)

        if i % print_freq == 0:
            print("Trajectory {}| Avg. Reward {:.3f}".format(i, np.mean(rewards[-print_freq:])))

        if is_done.is_set():
            env.close()
            return

        while is_paused.is_set():
            time.sleep(2)
            if is_done.is_set():
                env.close()
                return
    env.close()


class Impala:
    def __init__(self, game:str, episodes: int, max_steps: int, stack_n_frames: int, n_action_repeats: int, print_freq: int):
        self.game = game
        self.env = EnvUtil.getEnvList([game])[0]
        self.episodes = episodes
        self.stack_n_frames = stack_n_frames
        self.n_action_repeats = n_action_repeats
        self.max_steps = max_steps
        self.print_freq = print_freq

        #impala specific hyperparameters
        self.trajectory_length = 40
        self.learning_rate = 0.0006
        self.gamma = 0.99
        self.batch_size = 32
        self.actor_num = 15
        self.learner_num = 1

        #global objects
        self.p_network = NeuralNet(self.stack_n_frames, self.env.action_space.n)
        self.v_network = NeuralNet(self.stack_n_frames, self.env.action_space.n)
        self.trajectory_queue = mp.Queue()
        self.is_done = mp.Event()
        self.is_done.clear()
        self.is_paused = mp.Event()
        self.is_paused.clear()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return 'IMPALA'

    def pause(self, event):
        self.is_paused.set()

        print("\nPaused")
        print("input \"1\" to quit, input anything else to continue (that doesn't start with \'p\')")
        choice = input("Enter your choice: ")

        # this is stupid, thank you input buffer
        choice = choice.lstrip("p")

        if choice == "1":
            self.save()
            self.is_done.set()
            return
        print("resuming")
        self.is_paused.clear()

    def run(self):

        keyboard.on_press_key('p', self.pause)

        rewards = []
        for i in range(self.episodes):  # loop per episode
            # episode setup
            e_reward = 0
            frame, info = self.env.reset()
            preprocessor = Preprocessor(frame, self.stack_n_frames)
            obs = preprocessor.get_obs(frame)
            lives = info['lives']

            for s in range(self.max_steps):  # loop per step
                # get action
                with torch.no_grad():
                    action_probs = self.p_network.forward(torch.from_numpy(obs).float().unsqueeze(0)).detach().numpy()
                    action = np.argmax(action_probs)


                # run action
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

                # thank you input buffer
                choice = choice.lstrip("p")

                if choice == "1":
                    self.save()
                    return
                print("resuming")
                self.paused = False

    def train(self):
        keyboard.on_press_key('p', self.pause)

        queues = [mp.Queue() for _ in range(self.actor_num)]
        actors = [mp.Process(target=actor, args=[self.game, self.trajectory_queue, queues[i], self.is_done, self.is_paused, self.trajectory_length, self.stack_n_frames, self.n_action_repeats]) for i in range(self.actor_num)]
        for a in actors:
            a.start()

        time.sleep(2)

        learner(self.game, self.trajectory_queue, queues, self.is_done, self.is_paused, self.trajectory_length, self.stack_n_frames, self.n_action_repeats, self.episodes, self.print_freq)
        print("\nTraining finished")
        self.is_done.set()

        for a in actors:
            a.join(timeout=1)
            if a.is_alive():
                a.terminate()

        return

    def save(self):
        return
    def load(self):
        return