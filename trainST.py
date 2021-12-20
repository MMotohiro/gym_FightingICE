import numpy as np
import time
import matplotlib.pyplot as plt
import os, sys
from collections import deque
from memory import Memory
from DQNAgent import NN, LSTM
import json, csv
import glob

MODEL_NAME = "param.LT01"
MODEL_PATH = "./model/" + MODEL_NAME
TRAIN_PATH = "./learningData/train/"
TEST_PATH = "./learningData/test/"


def main():
    action_size = 3
    rawData = None
    files = glob.glob(DATA_PATH +"*.csv")
    datas = []
    test_targets = [0] * 3
    
    #model init
    model = NN(action_size) 

    model.load__model(MODEL_PATH)
    model1.load_model(MODEL_PATH)



    # read csv
    for file in files:
        with open(file, 'r') as f:
            try:
                reader = csv.reader(f)
                datas.extend([row for row in reader])
            except:
                pass

    #json to numpy ndarray
    state_len = len(get_obs(datas[0]))
    inputs =  np.zeros((len(datas),state_len))
    targets = np.zeros((len(datas),action_size))
    print("data size:", len(datas))
    print("state len:", state_len)

    for i, data in enumerate(datas):
        if(data[-1] != "None"):
            #最初と最後だけでいいの
            downCount = [0,0] #39
            if(data[27] == "1"):
                downCount[0] += 1
            elif(data[92]== "1"):
                downCount[1] += 1
            damage = [0,0]
            damage[0] = temp[4] - data[0]
            damage[1] = temp[5] - data[65]

            hp = [0,0]
            hp = [data[0],data[65]]
            energy = [data[1],data[66]]
            
            myOppX = [data[2],data[67]]
            range = abs(data[2]-data[67])
            time = i - temp[11] 



            val = []
            val.extend(downCount)
            val.extend(damage)
            val.extend(hp)
            val.extend(energy)
            val.append(range)
            val.enxtend(myOppX)
            val.append(time)



        

    print(inputs[0:5])
    print(targets[0:5])
    print(test_targets)
    print("**************")
    print("* START TRAIN*")
    print("**************")

    


    # model.fit(inputs,targets)
    
    #model init
    # model = NNTuner(action_size, inputs,targets) 
    # model.fit()


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
    test_targets =[0] * 15

    temp = 5
    tempC = None
    timer = 0
    count = 0
    for i,data in enumerate(testDatas):
        act = get_target(data) 

        if(act == 4): #移動もしくはタイマーが0以上のとき
            temp = get_target(data)
            continue

        testInputs[count:count+1] = get_obs(data)
        testTargets[count][act] = 1
        # test_targets[act] += 1
        
        temp = get_target(data)
        count += 1
        

    inputs = inputs[:count]
    targets = targets[:count]
    

    for i,data in enumerate(testInputs):
        print(np.argmax(model.predict(data.reshape(1,len(data)))))
        print(np.argmax(model1.predict(data.reshape(1,len(data)))))
        # test_targets[np.argmax(model.predict(data.reshape(1,len(data))))] += 1
    
    los, acc = model.evaluate(testInputs, testTargets)
    print("loss=", los)
    print("acc =",acc)
    print(test_targets)
    model.save_model(MODEL_PATH)



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