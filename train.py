import gym
import time
from gym_fightingice.envs.Machete import Machete

from observer import Observer
from rolebaseAgent import RoleBaseAgent
from DQNAgent import DQNAgent
from trainer import Trainer

MODEL_NAME = "param.CYR01"
MODEL_PATH = "./model/" + MODEL_NAME

def main():
    gymEnv = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".", port=4242)
    # HACK: aciontから自動で取ってこれるようにしておく
    action_size = 56
    learning_rate = 0.001
    batch_size = 10
    episode = 1000
    gamma = 0.99
    greedy_value = 1.0

    p2 = "MctsAi"
    env = Observer(gymEnv, p2)
    agent = DQNAgent(learning_rate, action_size, greedy_value)
    # agent.model.load_model(MODEL_PATH)
    print("************\nload  model\n************")
    # agent = RoleBaseAgent()
    trainer = Trainer(env, agent)
    print("************\n Sarrt learning\n************")
    trainer.train(episode, batch_size, gamma)
    
    print("************\nall end\n************")
    gymEnv.close()
    exit()
    

if __name__ == "__main__":
    main()
