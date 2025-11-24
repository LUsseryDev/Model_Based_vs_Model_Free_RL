import time

import keyboard
import torch
import gymnasium as gym

import EnvUtil
from MiscUtil import *
import torch.multiprocessing as mp
from torch.nn import functional as F


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, action_space):
        super(NeuralNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(1),
            torch.nn.Linear(in_features=5632, out_features=256),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EnvUtil.getEnvList([game])[0]
    network = NeuralNet(stack_n_frames, env.action_space.n).to(device)


    frame, info = env.reset()
    preprocessor = Preprocessor(frame, stack_n_frames)
    obs = preprocessor.get_obs(frame)
    lives = info['lives']

    #main loop
    while True:
        obs_trajectory = []
        reward_trajectory = []
        action_trajectory = []
        logit_trajectory = []


        #update network
        if not mqueue.empty():
            network.load_state_dict(mqueue.get())

        #make a trajectory
        for i in range(trajectory_length):
            # get action
            with torch.no_grad():
                action_probs = network.forward(torch.from_numpy(obs).float().unsqueeze(0).to(device)).cpu().detach()
                try: action = torch.multinomial(torch.nn.functional.softmax(action_probs, dim=1), 1)
                except:
                    print("broke")
                    print(action_probs)
                    raise RuntimeError

            # run action and save to trajectory
            for j in range(n_action_repeats - 1):
                env.step(action)
            next_frame, reward, terminated, truncated, info = env.step(action)

            obs_trajectory.append(obs)
            reward_trajectory.append(clip_reward(reward))
            action_trajectory.append(action)
            logit_trajectory.append(action_probs)

            #ready for next step
            obs = preprocessor.get_obs(next_frame)
            if lives != info['lives'] or terminated or truncated:
                frame, info = env.reset()
                preprocessor = Preprocessor(frame, stack_n_frames)
                obs = preprocessor.get_obs(frame)
                lives = info['lives']

        obs_trajectory = np.stack(obs_trajectory)
        action_trajectory = np.stack(action_trajectory)
        reward_trajectory = np.stack(reward_trajectory)
        logit_trajectory = np.stack(logit_trajectory)

        tqueue.put({"obs":obs_trajectory, "action":action_trajectory, "reward":reward_trajectory, "logits":logit_trajectory})


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

def learner(v_network:NeuralNet, p_network:NeuralNet, game:str, tqueue: mp.Queue, mqueues: list[mp.Queue], is_done:mp.Event(), is_paused:mp.Event(), trajectory_length:int, stack_n_frames:int, n_action_repeats:int, episodes:int, print_freq:int, batch_size:int, gamma:float) -> (NeuralNet, NeuralNet):
    process = mp.current_process()
    print("starting learner with pid "+str(process.pid))
    env = EnvUtil.getEnvList([game])[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_network.to(device)
    p_network.to(device)
    l2_loss = torch.nn.MSELoss()

    #hyperparamters
    #move these somewhere else
    rho_threshold = 1
    c_threshold = 1
    value_loss_scaler = 1
    policy_loss_scaler = 1
    entropy_loss_scaler = 1

    optimizer = torch.optim.Adam(list(v_network.parameters()) + list(p_network.parameters()), lr=0.0006)

    rewards = []
    losses = []
    #main training loop
    for i in range(episodes):

        #create batch
        batch = []
        for _ in range(batch_size):
            #if queue empty, wait
            while tqueue.empty():
                time.sleep(1)
            batch.append(tqueue.get())

        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_logits = []

        for t in batch:
            batch_obs.append(t["obs"])
            batch_action.append(t["action"])
            batch_reward.append(t["reward"])
            batch_logits.append(t["logits"])

        obs = torch.from_numpy(np.stack(batch_obs)).float().to(device)
        action = torch.from_numpy(np.stack(batch_action)).long().to(device).squeeze(2)
        actor_logits = torch.from_numpy(np.stack(batch_logits)).to(device).squeeze(2)
        reward = torch.from_numpy(np.stack(batch_reward)).to(device).unsqueeze(2)

        rewards.append(torch.mean(reward.float()))



        #do some learning
        values = [v_network(obs.select(1, j)) for j in range(trajectory_length)]
        learner_logits = [p_network(obs.select(1, j)) for j in range(trajectory_length)]

        learner_logits = torch.stack(learner_logits, dim=1).to(device)

        #print("learner: " + str(learner_logits.select(1, 0).shape))

        #used to start recursive v-trace calcs
        next_v_trace = values[-1]
        v_traces = []
        values_plus1 = values.copy()
        values_plus1.append(values[-1])
        values_plus1 = torch.stack(values_plus1, dim=1).to(device)

        advantages = []

        #calculate V-trace targets and losses
        for t in reversed(range(trajectory_length)):
            ratio = F.log_softmax(learner_logits, dim=-1).select(1, t).gather(1, action.select(1, t)) - F.log_softmax(actor_logits, dim=-1).select(1, t).gather(1, action.select(1, t))
            ratio = torch.exp(ratio)
            rho_t = torch.clamp(ratio, max=rho_threshold)
            cs = torch.clamp(ratio, max=c_threshold)
            delta = rho_t *(reward.select(1,t) + gamma*values_plus1.select(1, t+1) - values_plus1.select(1, t))
            v_traces.append(values_plus1.select(1,t) + delta + gamma*cs*(next_v_trace - values_plus1.select(1,t)))

            advantages.append(rho_t *(reward.select(1,t) + gamma*next_v_trace - values_plus1.select(1, t)))

            next_v_trace = v_traces[-1]

        v_traces.reverse()
        advantages.reverse()
        advantages = torch.stack(advantages, dim=1).to(device)
        value_tensor = torch.stack(values, dim=1).float().to(device)
        v_trace_tensor = torch.stack(v_traces, dim=1).float().to(device)

        #print("advantages {}".format(advantages.shape))

        #calculate losses (probably redo all of this, it sucks)
        #value loss
        value_loss = l2_loss(value_tensor, v_trace_tensor)

        #policy loss
        flat_action = torch.flatten(action, 0, 1).squeeze()
        flat_learner_logits = torch.flatten(learner_logits, 0, 1)
        cross_entropy = F.nll_loss(F.log_softmax(flat_learner_logits, dim=-1), target=flat_action, reduction="none")
        cross_entropy = cross_entropy.view_as(advantages)
        policy_loss = torch.sum(cross_entropy * advantages.detach())

        #entropy loss
        entropy_loss = 0.0006 * torch.sum(F.softmax(learner_logits, dim=-1) * F.log_softmax(learner_logits, dim=-1))


        total_loss = value_loss * value_loss_scaler + policy_loss * policy_loss_scaler + entropy_loss * entropy_loss_scaler

        losses.append(total_loss.item())



        #update parameters
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(p_network.parameters()) + list(v_network.parameters()), 40)
        optimizer.step()

        #update actors (this should have its own hyperparameter
        if i % 2 == 0:
            for q in mqueues:
                while not q.empty():
                    try: q.get_nowait()
                    except: pass
                q.put(p_network.state_dict())




        if i % print_freq == 0:
            print("Trajectory {} | T-queue size {} | Avg. Loss {} | Avg. Reward {:.3f}".format(i, tqueue.qsize(), np.mean(losses[-print_freq:]), torch.mean(torch.stack(rewards[-print_freq:]))))
            #print("Value Loss {} | Policy Loss {} | Entropy Loss {}".format(value_loss.item(), policy_loss.item(), entropy_loss.item()))

        if is_done.is_set():
            env.close()
            return p_network, v_network

        while is_paused.is_set():
            time.sleep(2)
            if is_done.is_set():
                env.close()
                return p_network, v_network
    env.close()
    return p_network, v_network


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
        self.actor_num = 10
        self.learner_num = 1
        self.max_queue_size = 1000

        #global objects
        self.p_network = NeuralNet(self.stack_n_frames, self.env.action_space.n)
        self.v_network = NeuralNet(self.stack_n_frames, 1)
        self.trajectory_queue = mp.Queue(maxsize=self.max_queue_size)
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

        self.p_network, self.v_network = learner(self.v_network, self.p_network, self.game, self.trajectory_queue, queues, self.is_done, self.is_paused, self.trajectory_length, self.stack_n_frames, self.n_action_repeats, self.episodes, self.print_freq, self.batch_size, self.gamma)
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