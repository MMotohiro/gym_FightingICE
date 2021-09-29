import gym
import time
from gym_fightingice.envs.Machete import Machete

from observer import Observer
from rolebaseAgent import RoleBaseAgent
from DQNAgent import DQNAgent
from trainer import Trainer

def main():
    gymEnv = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".", port=4242)
    # HACK: aciontから自動で取ってこれるようにしておく
    action_size = 56
    learning_rate = 0.1
    batch_size = 10
    episode = 50
    gamma = 0.99
    greedy_value = 0.4

    p2 = "MctsAi"
    env = Observer(gymEnv, p2)
    agent = DQNAgent(learning_rate, action_size, greedy_value)
    # agent.model.load_model('param.hdf6')
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
