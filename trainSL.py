import numpy as np
import time
import matplotlib.pyplot as plt
import os, sys
from collections import deque
from memory import Memory
from DQNAgent import NN, NNLSTM
import json, csv
import glob

MODEL_NAME = "param.SL01"
MODEL_PATH = "./model/" + MODEL_NAME
TRAIN_PATH = "./learningData/train/"
TEST_PATH = "./learningData/test/"


def main():
    action_size = 15
    gamma = 0.85
    greedy_value = 1.0
    rawData = None
    files = glob.glob(TRAIN_PATH +"*.json")
    datas = []
    test_targets = [0] * 15

    # read json
    for file in files:
        with open(file, 'r') as f:
            try:
                rawData = json.load(f)
                datas.extend(rawData["rounds"][0])
            except:
                pass
    
    #model init
    model = NN(action_size)    

    #json to numpy ndarray
    state_len = len(get_obs(datas[0]))
    inputs =  np.zeros((len(datas),state_len))
    targets = np.zeros((len(datas),action_size))
    print("data size:", len(datas))
    print("state len:", state_len)

    for i,data in enumerate(datas):
        inputs[i:i+1] = get_obs(data)
        targets[i][get_target(data)] = 1
        test_targets[get_target(data)] += 1

    print(inputs[0:5])
    print(targets[0:5])
    print(test_targets)
    model.fit(inputs,targets)

    #test
    files = glob.glob(TEST_PATH +"*.json")
    testDatas = []

    # read json
    for file in files:
        with open(file, 'r') as f:
            try:
                rawData = json.load(f)
                testDatas.extend(rawData["rounds"][0])
            except:
                pass

    state_len = len(get_obs(testDatas[0]))
    testInputs =  np.zeros((len(testDatas),state_len))
    testTargets = np.zeros((len(testDatas),action_size))
    los, acc = model.evaluate(testInputs, testTargets)
    print("loss=", los)
    print("acc =",acc)
    model.save_model(MODEL_PATH)
# スティック1~9とボタンA~Tを、0から14に割り当てる
# スティック1~9 → 0~8
# ボタンA~T → 9~14

def get_target(data):
    my = data["P1"]
    
    #ボタン処理
    button = [my["key_a"] , my["key_b"] , my["key_c"] , my["key_e"] , my["key_s"] , my["key_t"] ]
    for i in button:
        if(i):
            return button.index(True) + 9

    #移動処理
    stick = [my["key_up"], my["key_down"], my["key_left"], my["key_right"]]

    if(stick[2]):
        if(stick[0]):
            return 6
        elif(stick[1]):
            return 0
        else:
            return 3
    elif(stick[3]):
        if(stick[0]):
            return 8
        elif(stick[1]):
            return 2
        else:
            return 5
    elif(stick[0]):
        return 7
    elif(stick[1]):
        return 1
    
    return 4

def get_obs(data):
    my = data["P1"]
    opp = data["P2"]

    # my information
    myHp = abs(my["hp"] / 400)
    myEnergy = my["energy"] / 300
    myX = (my["left"] + my["right"] / 2) / 960
    myY = (my["top"] + my["bottom"] / 2) / 640
    mySpeedX = my["speed_x"] / 15
    mySpeedY = my["speed_y"] / 28
    myState = my["state_id"]
    myRemainingFrame = my["remaining_frames"] / 70

    # opp information
    oppHp = abs(opp["hp"] / 400)
    oppEnergy = opp["energy"] / 300
    oppX = (opp["left"] + opp["right"] / 2) / 960
    oppY = (opp["top"] + opp["bottom"] / 2) / 640
    oppSpeedX = opp["speed_x"] / 15
    oppSpeedY = opp["speed_y"] / 28
    oppState = opp["state_id"]
    oppRemainingFrame = opp["remaining_frames"] / 70

    # time information
    game_frame_num = data["current_frame"] / 3600

    observation = []

    # my information
    observation.append(myHp)
    observation.append(myEnergy)
    observation.append(myX)
    observation.append(myY)
    if mySpeedX < 0:
        observation.append(0)
    else:
        observation.append(1)
    observation.append(abs(mySpeedX))
    if mySpeedY < 0:
        observation.append(0)
    else:
        observation.append(1)
    observation.append(abs(mySpeedY))
    for i in range(56):
        if i == myState:
            observation.append(1)
        else:
            observation.append(0)
    observation.append(myRemainingFrame)

    # opp information
    observation.append(oppHp)
    observation.append(oppEnergy)
    observation.append(oppX)
    observation.append(oppY)
    if oppSpeedX < 0:
        observation.append(0)
    else:
        observation.append(1)
    observation.append(abs(oppSpeedX))
    if oppSpeedY < 0:
        observation.append(0)
    else:
        observation.append(1)
    observation.append(abs(oppSpeedY))
    for i in range(56):
        if i == oppState:
            observation.append(1)
        else:
            observation.append(0)
    observation.append(oppRemainingFrame)

    # time information
    observation.append(game_frame_num)

    myProjectiles = my['projectiles']
    oppProjectiles = opp['projectiles']

    if len(myProjectiles) == 2:
        myHitDamage = myProjectiles[0]["hit_damage"] / 20.0
        myHitAreaNowX = ((myProjectiles[0]["hit_area"]["left"] + myProjectiles[
            0]["hit_area"]["right"]) / 2) / 960.0
        myHitAreaNowY = ((myProjectiles[0]["hit_area"]["top"] + myProjectiles[
            0]["hit_area"]["bottom"]) / 2) / 640.0
        observation.append(myHitDamage)
        observation.append(myHitAreaNowX)
        observation.append(myHitAreaNowY)
        myHitDamage = myProjectiles[1]["hit_damage"] / 20.0
        myHitAreaNowX = ((myProjectiles[1]["hit_area"]["left"] + myProjectiles[
            1]["hit_area"]["right"]) / 2) / 960.0
        myHitAreaNowY = ((myProjectiles[1]["hit_area"]["top"] + myProjectiles[
            1]["hit_area"]["bottom"]) / 2) / 640.0
        observation.append(myHitDamage)
        observation.append(myHitAreaNowX)
        observation.append(myHitAreaNowY)
    elif len(myProjectiles) == 1:
        myHitDamage = myProjectiles[0]["hit_damage"] / 20.0
        myHitAreaNowX = ((myProjectiles[0]["hit_area"]["left"] + myProjectiles[
            0]["hit_area"]["right"]) / 2) / 960.0
        myHitAreaNowY = ((myProjectiles[0]["hit_area"]["top"] + myProjectiles[
            0]["hit_area"]["bottom"]) / 2) / 640.0
        observation.append(myHitDamage)
        observation.append(myHitAreaNowX)
        observation.append(myHitAreaNowY)
        for t in range(3):
            observation.append(0.0)
    else:
        for t in range(6):
            observation.append(0.0)

    if len(oppProjectiles) == 2:
        oppHitDamage = oppProjectiles[0]["hit_damage"] / 20.0
        oppHitAreaNowX = ((oppProjectiles[0]["hit_area"]["left"] + oppProjectiles[
            0]["hit_area"]["right"]) / 2) / 960.0
        oppHitAreaNowY = ((oppProjectiles[0]["hit_area"]["top"] + oppProjectiles[
            0]["hit_area"]["bottom"]) / 2) / 640.0
        observation.append(oppHitDamage)
        observation.append(oppHitAreaNowX)
        observation.append(oppHitAreaNowY)
        oppHitDamage = oppProjectiles[1]["hit_damage"] / 20.0
        oppHitAreaNowX = ((oppProjectiles[1]["hit_area"]["left"] + oppProjectiles[
            1]["hit_area"]["right"]) / 2) / 960.0
        oppHitAreaNowY = ((oppProjectiles[1]["hit_area"]["top"] + oppProjectiles[
            1]["hit_area"]["bottom"]) / 2) / 640.0
        observation.append(oppHitDamage)
        observation.append(oppHitAreaNowX)
        observation.append(oppHitAreaNowY)
    elif len(oppProjectiles) == 1:
        oppHitDamage = oppProjectiles[0]["hit_damage"] / 20.0
        oppHitAreaNowX = ((oppProjectiles[0]["hit_area"]["left"] + oppProjectiles[
            0]["hit_area"]["right"]) / 2) / 960.0
        oppHitAreaNowY = ((oppProjectiles[0]["hit_area"]["top"] + oppProjectiles[
            0]["hit_area"]["bottom"]) / 2) / 640.0
        observation.append(oppHitDamage)
        observation.append(oppHitAreaNowX)
        observation.append(oppHitAreaNowY)
        for t in range(3):
            observation.append(0.0)
    else:
        for t in range(6):
            observation.append(0.0)

    observation = np.array(observation, dtype=np.float32)
    observation = np.clip(observation, 0, 1)
    return observation
    
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