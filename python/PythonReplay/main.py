import sys, os, datetime
import msvcrt
from threading import (Event, Thread)
from time import sleep
from ltSurvey import ltSurvery
import tkinter as tk
from multiprocessing import Process, Pipe
import copy

sys.path .append('../')
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field


#LIST OF EMOTION
KEY_LIST = {'1':"h0", '4':"h1", '7':"h2", '2':"a0", '5':"a1", '8':"a2", '3':"s0", '6':"s1", '9':"s2",}
REPLAY_NAME = "HPMode_KeyBoard_SimpleAI_2021.10.27-16.39.35"
DOUNW_TIMER = 120

def main():
    ep1,ep2 = Pipe()
    th1 = Process(target=replay, args=(ep1,))
    th2 = Process(target=emotionGUI, args=(ep2,))
    th1.start()
    th2.start()
    

def replay(ep):
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242), callback_server_parameters=CallbackServerParameters());
    manager = gateway.entry_point
    args = sys.argv

    dt_now = datetime.datetime.now()

    REPLAY_NAME = "HPMode_KeyBoard_SimpleAI_2021.10.27-16.39.35"
    # get replay path
    try:
        REPLAY_NAME = args[1]
    except:
        print("replay file is not found")
        pass

    LOG_NAME = REPLAY_NAME + ".csv"
    LOG_PATH = "./logs/" + LOG_NAME
    LT_PATH =  "./logs/lt_" + LOG_NAME
    #init replay
    print("Replay: Loading")
    replay = manager.loadReplay(REPLAY_NAME) # Load replay data

    print("Replay: Init")
    replay.init()
    e_list = []
    lt_list = []
    r_temp = 1

    sleep(5)

    DOUNW_TIMER = 120
    downCounter = DOUNW_TIMER

    # Main process
    for i in range(12000): # Simulate 100 frames
        # print("Replay: Run frame", i)    
        framedata = replay.getFrameData()
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key != b'\xe0':
                key = key.decode()
            else:
                key = msvcrt.getch()

            if key == 'q':
                break


        sys.stdout.flush()
        replay.updateState()
        # lt servey
        r_num = framedata.getRound()


        # 自分もしくは敵のダウンを検知
        downCounter -= 1
        if(not framedata.getCharacter(True) is None):
            selfState = framedata.getCharacter(True).getState()
            oppState = framedata.getCharacter(False).getState()
            if((selfState.name() == "DOWN" or oppState.name() == "DOWN") and downCounter < 0):
                print("DOWN!")
                downCounter = DOUNW_TIMER
                val = ep.recv()



                

        # if r_temp != r_num and r_num >= 2:
        #     r_temp = r_num
        #     lt_temp = ltSurvery()
        #     lt_list.append(lt_temp)
            


        if replay.getState().name() == "CLOSE":
            break
        
    # end replay
    print("Replay: Close")
    print(e_list)

    #dataset with dataframe

    txt_list = []
    for i in e_list:
        txt_list.append(",".join(map(str, i)))

    with open(LOG_PATH, mode='w') as f:
        f.writelines(txt_list)

    # with open(LT_PATH, mode='w') as f:
    #     f.writelines(lt_list)

    replay.close()

    sys.stdout.flush()

    gateway.close_callback_server()
    gateway.close()



#                      #
#                      #
#      UI   BLOCK      #
#                      #
#                      #

def button_clk(ep,val):
    def inner():
        print(val)
        ep.send(val)
    return inner

def emotionGUI(ep):
    GUI_WIDTH = 960
    GUI_HIGTH = 300
    EMOTION_LIST = ["恐怖", "怒り", "悲しい", "嫌悪", "喜び"]

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
        func = button_clk(ep,copy.copy(emo+"1"))
        Button[0][-1].config(command=func)
        Button[0][-1].place(x=GUI_WIDTH // 6 * (i+1) - 20, y = GUI_HIGTH //6 * 4)
        #弱い感情
        Button[1].append(tk.Button(text=emo,width=8, height = 2, font=("", 20)))
        func = button_clk(ep,emo+"0")
        Button[1][-1].config(command=func)
        Button[1][-1].place(x=GUI_WIDTH // 6 * (i+1) - 20, y = GUI_HIGTH //6 * 2)


    root.mainloop()



if __name__ == "__main__":
    main()