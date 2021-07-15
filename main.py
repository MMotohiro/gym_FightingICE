import gym
import sys

sys.path.append('gym-fightingice')
from gym_fightingice.envs.Machete import Machete

def main():
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".", port=4242)
    env.reset(p2=Machete)
    observation = env.reset()
    for i in range(300):
        env.step(31)




if __name__ == "__main__":
    main()
