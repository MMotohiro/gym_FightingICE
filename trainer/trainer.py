import numpy as np
import matplotlib.pyplot as plt

from memory import Memory

class Trainer(object):
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
        self.memory = None

    def train(self, episode: int, batch_size: int, gamma: float):
        """
        学習を実施する

        :param episode: 試合数
        :param batch_size: experience replayを実施するときに用いる過去のデータ数
        :param gamma: 価値関数の値をどれだけ重要視するかどうか
        """

        reward_list = []

        for i in range(episode):

            frame_data = self.env.reset()
            # NOTE: 学習出来るように変形しておく
            frame_data = self.env.flatten(frame_data)
            done = False
            self.memory = Memory()
            state_len = len(frame_data[0])

            while not done:
                # TODO: 毎回get_observation_spaceを実行しないようにしておく
                action = self.agent.get_action(frame_data, self.env.get_observation_space())
                next_frame_data, reward, done, info = self.env.step(action)

                # NOTE: 学習出来るように変形しておく
                next_frame_data = self.env.flatten(next_frame_data)

                # NOTE: experience replayを実施するため試合を回しながら学習させない
                self.memory.add((frame_data, action, reward, next_frame_data))

                frame_data = next_frame_data

            batch = self.memory.sample(batch_size)

            # NOTE: 学習させるときにenvを変形させる. その時のenvのlenを入れる
            # FIXME: envのlenの管理方法を考える
            inputs = np.zeros((batch_size, state_len))
            targets = np.zeros((batch_size, self.agent.action_size))

            # ランダムに取り出した過去の行動記録から学習を実施(=experience replay)
            for j, (frame_data, action, reward, next_frame_data) in enumerate(batch):
                inputs[j: j+1] = frame_data

                expect_Q = self.agent.model.predict(next_frame_data)[0]
                # HACK: numpyに置き換える
                next_action = np.argmax(expect_Q)
                target = reward + gamma * self.agent.model.predict(next_frame_data)[0][next_action]

                # TODO: 理論を理解する
                targets[j] = self.agent.model.predict(frame_data)[0]
                targets[j][action] = target

            self.agent.update(inputs, targets)

            # NOTE: 試合が終了した際の敵と味方のHPの差を保存する
            last_frame = self.memory.get_last_data()[0]
            reward_list.append(last_frame[0][0] - last_frame[0][11])

        self.agent.model.save_model('param.hdf5')
        self.create_image(reward_list, 'reward.png')

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