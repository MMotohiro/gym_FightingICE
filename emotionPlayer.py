import numpy as np
import time
import matplotlib.pyplot as plt
import os, sys
import msvcrt
from action import Action
from DQNAgent import NN_SL

class EmotionPlayer(object):
    """ 選手の学習や試合の状態を管理する """

    # HACK: agentの基底クラスを実装してアップキャストする
    #       ルールベースAIも読み込めるようにしておく
    def __init__(self, env: any, agent: any):
        """
        初期化

        :param env: gymの環境(observerでwrap済み)
        :param agent: 学習させたいagent

        """
        self.env = env
        self.agent = agent
        self.actLog = None
        self.EMOTION_TIMER = 30

    def play(self, episode: int):
        """
        学習済みモデルを用いてプレイする
        """
        for i in range(episode):
            
            frame_data = self.env.reset()
            frame_data = self.env.flatten(frame_data)
            done = False
            observation_s =self.env.get_observation_space()
            total_step = 0

            print("*************")
            print("start round " + str(i) +"/"+str(episode))
            print("*************")
            timer = self.EMOTION_TIMER
            act_log = [0]*15
            while not done:
                print(self.agent.get_emotion())
                total_step += 1

                action = self.agent.get_action(frame_data)
                # action = self.model.predict(frame_data)
                # アクションの実行、記録
                next_frame_data, reward, done, info = self.env.step(action)
                # next_frame_data, reward, done, info = self.env.step(Action(action+1).name)

                if(timer <= 0):
                    timer = self.EMOTION_TIMER
                    b_emotion = self.agent.get_emotion()
                    self.agent.emotion_calc(next_frame_data)
                    a_emotion = self.agent.get_emotion()
                    print("*******************\n   emotion calc!\n*******************")

                # NOTE: 学習出来るように変形しておく
                next_frame_data = self.env.flatten(next_frame_data)
                frame_data = next_frame_data

                timer -= 1
                if msvcrt.kbhit():
                    if msvcrt.getch().decode() == 'q':
                        print("game interruption")
                        self.env.close()
                        exit()

            print(act_log)
            print(act_log[:9])
            print(act_log[9:])
            print("end round" + str(i+1) + "/" + str(episode))
            print("total step:"  + str(total_step))
