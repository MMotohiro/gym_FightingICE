import numpy as np
import time
import matplotlib.pyplot as plt
import os, sys
from collections import deque
from memory import Memory

from action import Action

class Trainer(object):
    """ 選手の学習や試合の状態を管理する """

    # HACK: agentの基底クラスを実装してアップキャストする
    #       ルールベースAIも読み込めるようにしておく
    def __init__(self, env: any, agent: any, model_name: str):
        """
        初期化

        :param env: gymの環境(observerでwrap済み)
        :param agent: 学習させたいagent

        """
        self.model_name = model_name
        self.model_path = "./model/" + self.model_name
        self.model_txt = "./model/" + self.model_name.split('.')[1] + ".txt"
        self.env = env
        self.agent = agent
        self.memory = None
        self.actLog = None


    def train(self, episode: int, batch_size: int, gamma: float):
        """
        学習を実施する

        :param episode: 試合数
        :param batch_size: experience replayを実施するときに用いる過去のデータ数
        :param gamma: 価値関数の値をどれだけ重要視するかどうか
        """
        print("START TRAIN")
        reward_list = []
        startEpisode = 0
        try:
            with open(self.model_txt ,'r') as f:
                startEpisode = int(f.read())   
                print("start from " + str(startEpisode))
        except:
            f = open(self.model_txt , 'w')
            f.write('0')
            f.close()
    
        print("MODEL LOADED")
        for i in range(startEpisode, episode):
            frame_data = self.env.reset()
            self.actLog = deque([38]*10,maxlen=10)
            # NOTE: 学習出来るように変形しておく
            frame_data = self.env.flatten(frame_data)
            done = False
            self.memory = Memory()
            state_len = len(frame_data[0])
            total_step = 0
            observation_s =self.env.get_observation_space()

            print("*************")
            print("start round " + str(i) +"/"+str(episode))
            print("epsilon = " + str(max((1.0 - (i * 1.3 / episode))  * self.agent.greedy_value , 0.01)))
            print("*************")
            while not done:
                total_step += 1
                # TODO: 毎回get_observation_spaceを実行しないようにしておく
                action = self.agent.get_action(frame_data, observation_s, i ,episode)
                # アクションの実行、記録
                next_frame_data, reward, done, info = self.env.step(Action(action + 1).name)

                # 学習用報酬調整
                print(type(next_frame_data))    

                #rewardを正規化
                if(reward != 0):
                    reward = reward / 50

                self.actLog.append(action)
                # NOTE: 学習出来るように変形しておく
                next_frame_data = self.env.flatten(next_frame_data)
                # if(reward != 0):
                # NOTE: experience replayを実施するため試合を回しながら学習させない
                self.memory.add((frame_data, action, reward, next_frame_data))
                frame_data = next_frame_data

            print("end round")
            print("total memory:" + str(self.memory.len()))

            if(i>0 and i % 10 == 9):
                continue

            # NOTE: 報酬を逆伝達
            memory_temp = Memory()
            reward_gamma = 0.95
            reward_temp = 0
            for _ in range(self.memory.len()):
                _temp = self.memory.pop()
                _reward = _temp[2] + reward_temp * reward_gamma
                memory_temp.addLeft((_temp[0], _temp[1], _reward, _temp[3]))
                reward_temp = _reward
            
            self.memory = memory_temp

            batch = self.memory.sample(batch_size)
            # batch = self.memory.sample((self.memory.len() // 4 ) * 3 )

            # NOTE: 学習させるときにenvを変形させる. その時のenvのlenを入れる
            # FIXME: envのlenの管理方法を考える
            inputs = np.zeros((batch_size, state_len))
            targets = np.zeros((batch_size, self.agent.action_size))
            print("action_size:" + str( self.agent.action_size))
            print("学習開始")
            # ランダムに取り出した過去の行動記録から学習を実施(=experience replay)
            for j, (frame_data, action, reward, next_frame_data) in enumerate(batch):
                inputs[j: j+1] = frame_data

                expect_Q = self.agent.model.predict(next_frame_data)[0]
                # HACK: numpyに置き換える
                next_action = np.argmax(expect_Q)
                target = reward + gamma * expect_Q[next_action]

                # TODO: 理論を理解する
                targets[j] = self.agent.model.predict(frame_data)[0]
                targets[j][action - 1] = target


                    
            print("更新開始")
            self.agent.update(inputs, targets)
            # NOTE: 試合が終了した際の敵と味方のHPの差を保存する
            print("記録")
            last_frame = self.memory.get_last_data()[0]
            reward_list.append(last_frame[0][0] - last_frame[0][11])

            
            print("end round" + str(i+1) + "/" + str(episode))
            print("total step:"  + str(total_step))
            # 5試合ごとにモデルを保存
            if(i % 5 == 0 and i > 0):
                print("save model....")
                self.agent.model.save_model(self.model_path+"-"+str(i))
                print("done saving model!")

                # 現在のepisode数を記録
                with open(self.model_txt ,'w') as f:
                    f.write(str(i)) 

        self.agent.model.save_model(self.model_txt)
        print("************\nmode save end\n************")
        self.create_image(reward_list, 'reward.png')
        print("************\ncreate image end\n************")

    # HACK: anyを許さない
    def create_image(self, data: any, image_path: str) -> None:
        """
        グラフを生成する

        :param data: グラフにプロットしたいデータ
        :param image_path: 画像の保存先
        """

        plt.clf()
        x = range(len(data))
        plt.plot(x, data)
        plt.savefig(image_path)