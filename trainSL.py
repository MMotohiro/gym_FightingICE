import numpy as np
import time
import matplotlib.pyplot as plt
import os, sys
from collections import deque
from memory import Memory
from DQNAgent import NN
import csv

MODEL_NAME = "param.MOD05"
MODEL_PATH = "./model/" + MODEL_NAME

def main():
    action_size = 20
    batch_size = 500
    episode = 500
    gamma = 0.85
    greedy_value = 1.0

    model = NN(action_size)

    # read csv
    with open() as f:
        reader = f.read()
        rawData = [row for row in reader]
    #csv to numpy ndarray
    rawData = rawData[:][:-1]
    inputs =  np.array(rawData)
    targets = np.zeros(5,5)
    model.fit(inputs,targets)


    
def flatten(data,emotion = None) -> np.ndarray:
    """
    NNに入力できるように配列に変形する

    :param data: 変形したいデータ
    :return: 変形後のデータ
    
    こちらも同じくCYR_AI形式に対応
    """
    if emotion != None:
        result = np.zeros((1, len(data)+1 ))
        result[0][-1] = emotion
    else:
        result = np.zeros((1, len(data)))

    for i in range(len(data)):
        result[0][i] = data[i]
    
    return result

if __name__ == "__main__":
    main()