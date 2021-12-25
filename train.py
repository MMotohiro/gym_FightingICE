import gym
import time
from gym_fightingice.envs.Machete import Machete

from observer import Observer
from rolebaseAgent import RoleBaseAgent
from DQNAgent import DQNAgent
from trainer import Trainer

MODEL_NAME = "param.MOD06"
MODEL_PATH = "./model/" + MODEL_NAME

def main():
    gymEnv = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".", port=4242)
    # gymEnv = gym.make("FightingiceDataNoFrameskipNd-v0", java_env_path=".", port=4242)
    # HACK: aciontから自動で取ってこれるようにしておく
    action_size = 21
    batch_size = 500
    episode = 500
    gamma = 0.85
    greedy_value = 1.0

    p2 = "MctsAi"
    p2 = RoleBaseAgent
    env = Observer(gymEnv, p2)
    agent = DQNAgent(action_size, greedy_value)
    try:
        print("found model data.\nloading....")
        agent.model.load_model(MODEL_PATH)
    except:
        pass
    print("************\nload  model\n************")
    # agent = RoleBaseAgent()
    trainer = Trainer(env, agent, MODEL_NAME)
    print("************\n Sarrt learning\n************")
    trainer.train(episode, batch_size, gamma)
    
    print("************\nall end\n************")
    gymEnv.close()
    exit()
    

if __name__ == "__main__":
    main()
