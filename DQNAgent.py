from typing import Dict, List, Union
from gym import spaces
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import numpy as np
import random

from action import Action
from state import State


def huberloss(y_true, y_pred) -> float:
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
    def __init__(self, action_size: int) -> None:
        """
        NNの初期化をやる

        :param action_size: 実施出来る行動の数
        """

        # HACK: モデルの層の構成を簡単に変更出来るようにしておく
        # HACK: 途中のデータ数を決め打ちしないようにする

        self.model = Sequential()

        #cyr Ai
        self.model.add(Dense(300, activation='relu', input_dim=143))
        self.model.add(Dense(300, activation='relu'))
        self.model.add(Dense(action_size, activation='softmax'))
        # self.model.compile(loss=huberloss, optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    # TODO: 入力データの型を決める
    def fit(self, data: any, label: any) -> None:
        """
        学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        self.model.fit(data, label, batch_size=32, epochs=128, validation_split=0.1)

    def predict(self, data: any) -> List[float]:
        """
        現在の状態から最適な行動を予想する

        :param data: 入力(現在の状態)
        """

        # NOTE: 出力値はそれぞれの行動を実施すべき確率
        # HACK: 整形部分はここでやりたくない
        return self.model.predict(data)
    
    def evaluate(self, data: any, label: any):
        return self.model.evaluate(data,label)

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

# HACK: NNを別ファイルに分離させてもいい
class NNTuner(object):
    """ 状態価値関数を予想する """
    def __init__(self, action_size: int, train_x, train_y) -> None:
        """
        NNの初期化をやる

        :param action_size: 実施出来る行動の数
        """
        self.action_size = action_size
        self.model = None
        self.train_x = train_x
        self.train_y = train_y
    
    # TODO: 入力データの型を決める
    def fit(self) -> None:
        """
        学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        study = optuna.create_study()
        study.optimize(self.objective, n_trials=100)
        print(study.best_params)

    def predict(self, data: any) -> List[float]:
        """
        現在の状態から最適な行動を予想する

        :param data: 入力(現在の状態)
        """

        # NOTE: 出力値はそれぞれの行動を実施すべき確率
        # HACK: 整形部分はここでやりたくない
        return self.model.predict(data)
    
    def evaluate(self, data: any, label: any):
        return self.model.evaluate(data,label)

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

    def objective(self, trial):
        print("call objective")

        #学習用データのコピー
        train_x_copy = self.train_x
        train_y_copy = self.train_y

        # #最適化するパラメータの設定

        #FC層のユニット数
        mid_units1 = int(trial.suggest_discrete_uniform("mid_units1", 100, 500, 100))
        mid_units2 = int(trial.suggest_discrete_uniform("mid_units2", 100, 500, 100))

        #optimizer
        optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])


        #cyr Ai
        model = Sequential()
        model.add(Dense(mid_units1, activation='relu', input_dim=143))
        model.add(Dense(mid_units2, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        # self.model.compile(loss=huberloss, optimizer='adam', metrics=['accuracy'])
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        print("model compile")
        history = model.fit(train_x_copy, train_y_copy, verbose=0, epochs=10, batch_size=128, validation_split=0.1)

        #セッションのクリア
        K.clear_session()
        del model, optimizer, mid_units1, mid_units2, train_x_copy, train_y_copy
        

        #検証用データに対する正答率が最大となるハイパーパラメータを求める
        return 1 - history.history["val_accuracy"][-1]

class NNLSTM(object):
    """ 状態価値関数を予想する """
    def __init__(self, action_size: int) -> None:
        """
        NNの初期化をやる

        :param action_size: 実施出来る行動の数
        """

        # HACK: モデルの層の構成を簡単に変更出来るようにしておく
        # HACK: 途中のデータ数を決め打ちしないようにする

        self.model = Sequential()

        #cyr Ai
        self.model.add(LSTM(200,batch_input_shape = (None, 10, action_size)))
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dense(action_size, activation='softmax'))
        # self.model.compile(loss=huberloss, optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    # TODO: 入力データの型を決める
    def fit(self, data: any, label: any) -> None:
        """
        学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        self.model.fit(data, label, batch_size=32,validation_split=0.1, epochs=20, shuffle=False)

    def predict(self, data: any) -> List[float]:
        """
        現在の状態から最適な行動を予想する

        :param data: 入力(現在の状態)
        """

        # NOTE: 出力値はそれぞれの行動を実施すべき確率
        # HACK: 整形部分はここでやりたくない
        return self.model.predict(data)
    
    def evaluate(self, data: any, label: any):
        return self.model.evaluate(data,label)

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


    def __init__(self, action_size: int, greedy_value: float) -> None:
        """
        初期化を実施

        :param action_size: 実施出来るアクションの数
        :param greedy_value: グリーディー法を実施するかどうかの確率
        """
        self.model = NN(action_size)
        self.action_size = action_size
        self.greedy_value = greedy_value

    def get_action(self, data: List[Union[int, float]], observation_space: spaces, episode: int, maxEpisode: int) -> Action:
        """
        現在の状態から最適な行動を求める
        :param data: 現在の状態
        :param observation_space: 画面のサイズなどの情報
        :return: 行動(int)
        """

        key = random.random()
        greedy_value = max((1.0 - (episode * 1.3 / maxEpisode))  * self.greedy_value , 0.01)

        if key < greedy_value:
            random_action_value = random.randint(0, self.action_size-1)
            return random_action_value
        action_value = self.model.predict(data)

        # NOTE: 一番評価値が高い行動を選択する(Actionにキャストしておく)
        # NOTE: +1しているのは列挙型が0ではなく1スタートだから
        best_action = np.argmax(action_value)

        return best_action


    def update(self, data: any, label: any) -> None:
        """
        選手の学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        self.model.fit(data, label)
