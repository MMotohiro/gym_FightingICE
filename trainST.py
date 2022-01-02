import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os, sys
from collections import deque
from memory import Memory
from sklearn.metrics import confusion_matrix
from DQNAgent import NN_emotion, NNTuner
import json, csv
import glob
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

MODEL_NAME = "param.LT05"
MODEL_PATH = "./model/" + MODEL_NAME
TRAIN_PATH = "./learningData/csv/"
TEST_PATH = "./learningData/test/"


def main():
    action_size = 3
    rawData = None
    files = glob.glob(TRAIN_PATH +"*.csv")
    datas = []
    test_targets = [0] * 3
    print("model init")
    #model init
    
    
    print("load csv")
    # read csv
    for file in files:
        with open(file, 'r') as f:
            try:
                reader = csv.reader(f)
                datas.extend([row + [i]  for i, row in enumerate(reader)])
            except:
                pass

    #json to numpy ndarray
    len_emotoin = 2
  
    inputs =  []
    targets = []
    emotion = []
    temp = [1] * 15
    timer = 0
    downCount = [0,0]
    flag = True

    for i, data in enumerate(datas[1:]):
        if(data[143] != "None"):
            if(data[-1] < timer):
                downCount = [0,0]
                flag = True
                temp = [1] * 15
            timer = data[-1]
            #最初と最後だけでいいの    
            if(data[27] == "1.0"):
                downCount[0] += 1
            elif(data[92]== "1.0"):
                downCount[1] += 1

            damage = [0,0]
            damage[0] = temp[2] - float(data[0])
            damage[1] = temp[3] - float(data[65])

            hp = [float(data[0]),float(data[65])]
            # energy = [float(data[1]),float(data[66])]
            
            myOppX = [float(data[2]),float(data[67])]
            # dist = abs(float(data[2])-float(data[67]))
            # time = i - temp[11] 
            if(flag):
                flag = False
                target_temp = target_converter(data[143])
                emotion = [target_temp[0], target_temp[1], target_temp[2]]* len_emotoin
                time = float(data[-1])

            down = [0.0, 0.0]
            if(data[27] == "1.0"):
                down[0] = 1.0
            elif(data[92] == "1.0"):
                down[1] = 1.0


            val = []
            # val.extend([downCount[0] / 10, downCount[1] / 10])
            val.extend(damage)
            val.extend(hp)
            # val.extend(energy)
            # val.append(dist)
            val.extend(myOppX)
            # val.append(time)
            val.extend(down)
            val.extend(emotion)
            
            rand_key = random.randint(0,2)
            if(target_converter(data[143])[0] == 1 and rand_key != 0):
                pass
            else:
                inputs.append(val)
                targets.append(target_converter(data[143]))

            emotion[3:].extend(target_converter(data[143]))
            # emotion = [].extend(target_converter(data[143]))
            temp = val


        
    inputs = np.array(inputs)
    targets = np.array(targets)
    print("data size:",len(inputs))
    print(inputs[0:20])
    print(targets[0:5])
    print(test_targets)

    model = NN_emotion(3) 
    # model = NNTuner(3,inputs,targets) 
    print("**************")
    print("* START TRAIN*")
    print("**************")
    model.fit(inputs,targets)
    
    #model init
    # model = NNTuner(action_size, inputs,targets) 
    # model.fit()

    predict_x = model.predict(inputs)
    predict_classes = predict_x.argmax(axis = 1)
    true_classes = targets.argmax(axis = 1)
    print("pred_x:",predict_classes)
    print("pred_y:",true_classes)
    sample_x = [0,0,0]
    sample_y = [0,0,0]
    for i in true_classes:
        sample_y[i]+=1
    for i in predict_classes:
        sample_x[i]+=1
    print(sample_x)
    print(sample_y)
    print(confusion_matrix(true_classes, predict_classes))

    model.save_model(MODEL_PATH)
    print_cmx(true_classes, predict_classes)
    return 0

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

    # 感情情報をone hotに変換する
    # [1,0,0] 楽しい
    # [0,1,0] 怒り
    # [0,0,1] 悲しみ
def target_converter(val):
    key = int(val)
    if(key == 0):
        return  [0,0,1]        
    elif(key == 1):
        return  [0,0,1]
    elif(key == 2):
        return [0,0,1]
    elif(key == 3):
        return [1,0,0]
    elif(key == 4):
        return [1,0,0]
    elif(key == 5):
        return [1,0,0]
    elif(key == 7):
        return [0,0,1]
    elif(key == 8):
        return [0,0,1]
    elif(key == 9):
        return [0,0,1]
    elif(key == 10):
        return [0,1,0]
    elif(key == 11):
        return [0,1,0]
    elif(key == 12):
        return [0,1,0]
    elif(key == 13):
        return [0,1,0]
    elif(key == 14):
        return [0,1,0]
    elif(key == 15):
        return [0,1,0]
    return [0,0,0]

# def target_converter(val):
#     key = int(val)
#     if(key == 0):
#         return  [0,0,0.5]        
#     elif(key == 1):
#         return  [0,0,0.8]
#     elif(key == 2):
#         return [0,0,1]
#     elif(key == 3):
#         return [0.5,0,0]
#     elif(key == 4):
#         return [0.8,0,0]
#     elif(key == 5):
#         return [1,0,0]
#     elif(key == 7):
#         return [0,0,1]
#     elif(key == 8):
#         return [0,0,1]
#     elif(key == 9):
#         return [0,0,1]
#     elif(key == 10):
#         return [0,0.5,0]
#     elif(key == 11):
#         return [0,0.8,0]
#     elif(key == 12):
#         return [0,1,0]
#     elif(key == 13):
#         return [0,1,0]
#     elif(key == 14):
#         return [0,1,0]
#     elif(key == 15):
#         return [0,1,0]
#     return [0,0,0]
    
def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cmx, annot=True, fmt='g' ,square = True)
    plt.show()
 



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