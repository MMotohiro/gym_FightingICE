from typing import Dict, List, Union
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random

from action import Action
from state import State

# HACK: any外す
def huberloss(y_true: any, y_pred: any) -> float:
    """
    損失関数に用いるhuber関数を実装
    参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py

    :param y_true: 正解データ
    :param y_pred: 予測データ
    :return: 誤差
    """
    err = y_true - y_pred
    cond = keras.backend.abs(err) < 1.0
    L2 = 0.5 * keras.backend.square(err)
    L1 = (keras.backend.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return keras.backend.mean(loss)

# HACK: NNを別ファイルに分離させてもいい
class NN(object):
    """ 状態価値関数を予想する """
    def __init__(self, learning_rate: float, action_size: int) -> None:
        """
        NNの初期化をやる

        :param learning_rate: 学習率
        :param action_size: 実施出来る行動の数
        """
        self.learning_rate = learning_rate

        # HACK: モデルの層の構成を簡単に変更出来るようにしておく
        # HACK: 途中のデータ数を決め打ちしないようにする

        self.model = Sequential()
        self.model.add(Dense(action_size, activation='relu', input_dim=35))
        self.model.add(Dense(action_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.model.compile(loss=huberloss, optimizer='adam')

    # TODO: 入力データの型を決める
    def fit(self, data: any, label: any) -> None:
        """
        学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        self.model.fit(data, label, epochs=1)

    def predict(self, data: any) -> List[float]:
        """
        現在の状態から最適な行動を予想する

        :param data: 入力(現在の状態)
        """

        # NOTE: 出力値はそれぞれの行動を実施すべき確率
        # HACK: 整形部分はここでやりたくない
        return self.model.predict(data)

    def save_model(self, model_path: str):
        """
        モデルを保存する

        :param model_path: 保存先のパス
        """
        self.model.save_weights(model_path)

    def load_model(self, model_path: str):
        """
        学習済みのモデルを読み込む

        :param model_path: 読み込みたいモデルのパス
        """
        self.model.load_weights(model_path)

class DQNAgent(object):
    """
    深層学習を用いて行動選択を行うエージェント
    """
    # TODO: モデルの保存や読み込み部分を実装する


    def __init__(self, learning_rate: float, action_size: int, greedy_value: float) -> None:
        """
        初期化を実施

        :param learning_rate: NNの学習率
        :param action_size: 実施出来るアクションの数
        :param greedy_value: グリーディー法を実施するかどうかの確率
        """
        self.model = NN(learning_rate, action_size)
        self.action_size = action_size
        self.greedy_value = greedy_value

    def get_action(self, data: List[Union[int, float]], observation_space: spaces) -> Action:
        """
        現在の状態から最適な行動を求める
        :param data: 現在の状態
        :param observation_space: 画面のサイズなどの情報
        :return: 行動(int)
        """

        key = random.random()
        if key < self.greedy_value:
            random_action_value = random.randint(0, self.action_size-2)
            return Action(random_action_value+1)

        action_value = self.model.predict(data)[0]

        # NOTE: 一番評価値が高い行動を選択する(Actionにキャストしておく)
        # NOTE: +1しているのは列挙型が1startから？
        best_action = Action(np.argmax(action_value)+1)

        return best_action


    def update(self, data: any, label: any) -> None:
        """
        選手の学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        self.model.fit(data, label)
