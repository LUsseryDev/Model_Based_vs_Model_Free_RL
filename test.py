import gymnasium as gym
import torch
from MiscUtil import *
from Algorithms.IMPALA import NeuralNet



if __name__ == '__main__':
    model = "impala2"
    game = "Pong"

    env = gym.make("ALE/"+game+"-v5", obs_type="grayscale", frameskip=1, render_mode="human")
    #env = gym.make("ALE/"+game+"-v5", obs_type="grayscale", frameskip=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = NeuralNet(4, env.action_space.n)
    network.load_state_dict(torch.load(model+"_"+game+".pt"))
    network.eval()
    network.to(device)

    rewards = []

    #episode num
    for e in range(1):
        episode_reward = 0
        frame, info = env.reset()
        preprocessor = Preprocessor(frame, 4)
        obs = preprocessor.get_obs(frame)
        for t in range(2000):

            with torch.no_grad():
                action_probs, _ = network.forward(torch.from_numpy(obs).float().unsqueeze(0).to(device))
                #print(action_probs.shape)
                action = torch.multinomial(torch.nn.functional.softmax(action_probs, dim=1), 1)
                #action = torch.argmax(action_probs, dim=1)
                #print(action)

            #action = env.action_space.sample()
            reward = 0
            for j in range(3):
                reward += env.step(action)[1]
            next_frame, r, terminated, truncated, info = env.step(action)
            reward += r

            episode_reward += reward

            obs = preprocessor.get_obs(next_frame)

            if terminated or truncated:
                break
        rewards.append(episode_reward)
        print("Episode "+str(e)+" reward: "+str(episode_reward))


    print("Average reward:", np.mean(rewards))
    print("Max reward:", np.max(rewards))



