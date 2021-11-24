import numpy as np
import time
import matplotlib.pyplot as plt
import os, sys
import msvcrt

class Player(object):
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

            while not done:
                total_step += 1
                action = self.agent.get_action(frame_data, observation_s, i ,episode)
                # アクションの実行、記録
                next_frame_data, reward, done, info = self.env.step(action)
                if(reward != 0):
                    print("reward:"+ str(reward))

                # NOTE: 学習出来るように変形しておく
                next_frame_data = self.env.flatten(next_frame_data)
                frame_data = next_frame_data

                if msvcrt.kbhit():
                    if msvcrt.getch().decode() == 'q':
                        print("game interruption")
                        self.env.close()
                        exit()

            print("end round" + str(i+1) + "/" + str(episode))
            print("total step:"  + str(total_step))
