import gym
import time
from gym_fightingice.envs.Machete import Machete

from observer import Observer
from rolebaseAgent import RoleBaseAgent
from DQNAgent import EmotionAgent
from player import Player

E_MODEL_NAME = "param.LT04"
N_MODEL_NAME = "param.SL04"
H_MODEL_NAME = "param.SL04"
A_MODEL_NAME = "param.ANG01"
S_MODEL_NAME = "param.SAD02"
E_MODEL_PATH = "./model/" + E_MODEL_NAME
N_MODEL_PATH = "./model/" + N_MODEL_NAME
H_MODEL_PATH = "./model/" + H_MODEL_NAME
A_MODEL_PATH = "./model/" + A_MODEL_NAME
S_MODEL_PATH = "./model/" + S_MODEL_NAME

def main():
    gymEnv = gym.make("FightingiceDataFrameskip-v0", java_env_path=".", port=4242)
    # HACK: aciontから自動で取ってこれるようにしておく
    action_size = 15
    episode = 3
    greedy_value = 0

    p2 = RoleBaseAgent
    env = Observer(gymEnv, p2)
    # env = Observer(gymEnv, "KeyBoard")
    agent = EmotionAgent(E_MODEL_PATH, N_MODEL_PATH, H_MODEL_PATH, A_MODEL_PATH, S_MODEL_PATH)
    player = Player(env, agent)

    print("************\n Sarrt playing\n************")
    player.play(episode)
    
    print("************\nall end\n************")
    gymEnv.close()
    exit()
    

if __name__ == "__main__":
    main()
