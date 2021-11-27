import sys, os, datetime
import numpy as np
import msvcrt
from threading import (Event, Thread)
from time import sleep
from ltSurvey import ltSurvery
import tkinter as tk
from multiprocessing import Process, Pipe
import copy
import csv

sys.path .append('../')
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field


#LIST OF EMOTION
KEY_LIST = {'1':"h0", '4':"h1", '7':"h2", '2':"a0", '5':"a1", '8':"a2", '3':"s0", '6':"s1", '9':"s2",}
REPLAY_NAME = "lastgame"
DOUNW_TIMER = 120

def main():
    ep1,ep2 = Pipe()
    th1 = Process(target=replay, args=(ep1,REPLAY_NAME))
    th2 = Process(target=emotionGUI, args=(ep2,))
    th1.start()
    th2.start()
    

def replay(ep, REPLAY_NAME):
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242), callback_server_parameters=CallbackServerParameters());
    manager = gateway.entry_point

    dt_now = datetime.datetime.now()
    LOG_NAME = REPLAY_NAME + ".csv"
    LOG_PATH = "./logs/" + LOG_NAME
    LT_PATH =  "./logs/lt_" + LOG_NAME

    e_list = []
    obs_list = []
    lt_list = []
    r_temp = 1

    #init replay
    print("Replay: Loading")
    replayF = manager.loadReplay(REPLAY_NAME) # Load replay data

    print("Replay: Init")
    replayF.init()

    DOUNW_TIMER = 120
    downCounter = DOUNW_TIMER

    # Main process
    for i in range(60000): 
        # print("Replay: Run frame", i)    
        if msvcrt.kbhit():
            if key == 'q':
                break
        
        framedata = replayF.getFrameData()
        # 自分もしくは敵のダウンを検知
        downCounter -= 1
        if(not framedata.getCharacter(True) is None):
            selfState = framedata.getCharacter(True).getState()
            oppState = framedata.getCharacter(False).getState()
            val = "None"
            if((selfState.name() == "DOWN" or oppState.name() == "DOWN") and downCounter < 0):
                print("DOWN!")
                downCounter = DOUNW_TIMER
                # val = ep.recv()
                val = "ok"
            
            obs = get_obs(framedata).tolist()
            obs.append(val)
            obs_list.append(obs)

            # 終了処理
            if(framedata.getCharacter(True).getHp() <= 0 or framedata.getCharacter(False).getHp() <= 0 or framedata.getRemainingTime() <= 0):
                break

        # if r_temp != r_num and r_num >= 2:
        #     r_temp = r_num
        #     lt_temp = ltSurvery()
        #     lt_list.append(lt_temp)
            
        sys.stdout.flush()
        replayF.updateState()


        
    # end replay
    print("Replay: Close")

    # txt_list = []
    # for i in e_list:
    # #     txt_list.append(",".join(map(str, i)))

    # with open(LOG_PATH, mode='w') as f:
    #     f.writelines(txt_list)

    print(obs_list[:5])
    with open(LOG_PATH, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(obs_list)

    # with open(LT_PATH, mode='w') as f:
    #     f.writelines(lt_list)

    replayF.close()
    sys.stdout.flush()

    gateway.close_callback_server()
    gateway.close()

def get_obs(frameData):
    my = frameData.getCharacter(True)
    opp = frameData.getCharacter(False)

    # my information
    myHp = abs(my.getHp() / 400)
    myEnergy = my.getEnergy() / 300
    myX = ((my.getLeft() + my.getRight()) / 2) / 960
    myY = ((my.getBottom() + my.getTop()) / 2) / 640
    mySpeedX = my.getSpeedX() / 15
    mySpeedY = my.getSpeedY() / 28
    myState = my.getAction().ordinal()
    myRemainingFrame = my.getRemainingFrame() / 70

    # opp information
    oppHp = abs(opp.getHp() / 400)
    oppEnergy = opp.getEnergy() / 300
    oppX = ((opp.getLeft() + opp.getRight()) / 2) / 960
    oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
    oppSpeedX = opp.getSpeedX() / 15
    oppSpeedY = opp.getSpeedY() / 28
    oppState = opp.getAction().ordinal()
    oppRemainingFrame = opp.getRemainingFrame() / 70

    # time information
    game_frame_num = frameData.getFramesNumber() / 3600

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

    myProjectiles = frameData.getProjectilesByP1()
    oppProjectiles = frameData.getProjectilesByP2()

    if len(myProjectiles) == 2:
        myHitDamage = myProjectiles[0].getHitDamage() / 200.0
        myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
            0].getCurrentHitArea().getRight()) / 2) / 960.0
        myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
            0].getCurrentHitArea().getBottom()) / 2) / 640.0
        observation.append(myHitDamage)
        observation.append(myHitAreaNowX)
        observation.append(myHitAreaNowY)
        myHitDamage = myProjectiles[1].getHitDamage() / 200.0
        myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
            1].getCurrentHitArea().getRight()) / 2) / 960.0
        myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
            1].getCurrentHitArea().getBottom()) / 2) / 640.0
        observation.append(myHitDamage)
        observation.append(myHitAreaNowX)
        observation.append(myHitAreaNowY)
    elif len(myProjectiles) == 1:
        myHitDamage = myProjectiles[0].getHitDamage() / 200.0
        myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
            0].getCurrentHitArea().getRight()) / 2) / 960.0
        myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
            0].getCurrentHitArea().getBottom()) / 2) / 640.0
        observation.append(myHitDamage)
        observation.append(myHitAreaNowX)
        observation.append(myHitAreaNowY)
        for t in range(3):
            observation.append(0.0)
    else:
        for t in range(6):
            observation.append(0.0)

    if len(oppProjectiles) == 2:
        oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
        oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
            0].getCurrentHitArea().getRight()) / 2) / 960.0
        oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
            0].getCurrentHitArea().getBottom()) / 2) / 640.0
        observation.append(oppHitDamage)
        observation.append(oppHitAreaNowX)
        observation.append(oppHitAreaNowY)
        oppHitDamage = oppProjectiles[1].getHitDamage() / 200.0
        oppHitAreaNowX = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[
            1].getCurrentHitArea().getRight()) / 2) / 960.0
        oppHitAreaNowY = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[
            1].getCurrentHitArea().getBottom()) / 2) / 640.0
        observation.append(oppHitDamage)
        observation.append(oppHitAreaNowX)
        observation.append(oppHitAreaNowY)
    elif len(oppProjectiles) == 1:
        oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
        oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
            0].getCurrentHitArea().getRight()) / 2) / 960.0
        oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
            0].getCurrentHitArea().getBottom()) / 2) / 640.0
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


#                      #
#                      #
#      UI   BLOCK      #
#                      #
#                      #

def button_clk(ep, flag, val):
    def inner():
        print(val)
        ep.send(val)
    return inner

def emotionGUI(ep):
    GUI_WIDTH = 960
    GUI_HIGTH = 300
    EMOTION_LIST = ["恐怖", "怒り", "悲しい", "嫌悪", "喜び"]

    flag = False
    root = tk.Tk()
    root.title("emotion survey")
    root.geometry(str(GUI_WIDTH)+"x"+str(GUI_HIGTH))

    #ラベル表示
    Label = tk.Label(text = "強\n(Strong)",width=8, height = 2, font=("", 20))
    Label.place(x=0, y = GUI_HIGTH //6 * 2)
    Label = tk.Label(text = "弱\n(Weak)",width=8, height = 2, font=("", 20))
    Label.place(x=0, y = GUI_HIGTH //6 * 4)
    Button = [[],[]]

    for i,emo in enumerate(EMOTION_LIST):
        # 強い感情
        Button[0].append(tk.Button(text=emo,width=8, height = 2, font=("", 20)))
        func = button_clk(ep, flag, copy.copy(emo+"1"))
        Button[0][-1].config(command=func)
        Button[0][-1].place(x=GUI_WIDTH // 6 * (i+1) - 20, y = GUI_HIGTH //6 * 4)
        #弱い感情
        Button[1].append(tk.Button(text=emo,width=8, height = 2, font=("", 20)))
        func = button_clk(ep, flag, emo+"0")
        Button[1][-1].config(command=func)
        Button[1][-1].place(x=GUI_WIDTH // 6 * (i+1) - 20, y = GUI_HIGTH //6 * 2)



    root.mainloop()



if __name__ == "__main__":
    main()