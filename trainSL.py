import numpy as np
import time
import matplotlib.pyplot as plt
import os, sys
from collections import deque
from memory import Memory
from DQNAgent import DQNAgent

MODEL_NAME = "param.MOD05"
MODEL_PATH = "./model/" + MODEL_NAME

def main():
    action_size = 20
    batch_size = 500
    episode = 500
    gamma = 0.85
    greedy_value = 1.0

    model = NN(action_size)
    
    

