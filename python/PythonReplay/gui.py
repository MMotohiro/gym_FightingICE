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

def main():
    ep1,ep2 = Pipe()
    # th1 = Process(target=replay, args=(ep1,REPLAY_NAME))
    th2 = Process(target=emotionGUI, args=(ep2,))
    # th1.start()
    th2.start()

def button_clk(ep, val):
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

    w = root.winfo_screenwidth()    #モニター横幅取得
    h = root.winfo_screenheight()   #モニター縦幅取得
    w = w - GUI_WIDTH                  #メイン画面横幅分調整
    h = h - GUI_HIGTH -100                    #メイン画面縦幅分調整

    root.title("emotion survey")
    root.geometry(str(GUI_WIDTH)+"x"+str(GUI_HIGTH) + "+" + str(w)+"+"+str(h) )

    #replay start
    button = tk.Button(text="START Replay",width=15, height = 1, font=("", 20))
    func = button_clk(ep, ["command", "start"])
    button.config(command=func)
    button.place(x=30, y = GUI_HIGTH //6 - 20 )

    #ラベル表示
    Label = tk.Label(text = "強\n(Strong)",width=8, height = 2, font=("", 20))
    Label.place(x=0, y = GUI_HIGTH //6 * 2)
    Label = tk.Label(text = "弱\n(Weak)",width=8, height = 2, font=("", 20))
    Label.place(x=0, y = GUI_HIGTH //6 * 4)
    Button = [[],[]]
    for i,emo in enumerate(EMOTION_LIST):
        # 強い感情
        Button[0].append(tk.Button(text=emo,width=8, height = 2, font=("", 20)))
        func = button_clk(ep, ["emotion", copy.copy(emo+"1")])
        Button[0][-1].config(command=func)
        Button[0][-1].place(x=GUI_WIDTH // 6 * (i+1) - 20, y = GUI_HIGTH //6 * 4)
        #弱い感情
        Button[1].append(tk.Button(text=emo,width=8, height = 2, font=("", 20)))
        func = button_clk(ep,  ["emotion", copy.copy(emo+"0")])
        Button[1][-1].config(command=func)
        Button[1][-1].place(x=GUI_WIDTH // 6 * (i+1) - 20, y = GUI_HIGTH //6 * 2)



    root.mainloop()



if __name__ == "__main__":
    main()