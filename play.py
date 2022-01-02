import gym
import time
from gym_fightingice.envs.Machete import Machete

from observer import Observer
from rolebaseAgent import RoleBaseAgent
from DQNAgent import DQNAgent, PAgent
from player import Player

MODEL_NAME = "param.TEST01"
MODEL_PATH = "./model/" + MODEL_NAME

def main():
    gymEnv = gym.make("FightingiceDataFrameskip-v0", java_env_path=".", port=4242)
    # HACK: aciontから自動で取ってこれるようにしておく
    action_size = 15 #15 or 21
    episode = 500
    greedy_value = 0

    p2 = RoleBaseAgent
    env = Observer(gymEnv, p2)
    # env = Observer(gymEnv, "KeyBoard")
    agent = PAgent(action_size)
    try:
        print("found model data.\nloading....")
        agent.model.load_model(MODEL_PATH)
        print("**success!**")
    except:
        print("Could not found model data")
        gymEnv.close()
        exit()
    player = Player(env, agent)

    print("************\n Sarrt playing\n************")
    player.play(episode)
    
    print("************\nall end\n************")
    gymEnv.close()
    exit()
    

if __name__ == "__main__":
    main()
