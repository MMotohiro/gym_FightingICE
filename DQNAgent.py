from typing import Dict, List, Union
from gym import spaces
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM
from tensorflow.keras.optimizers import Adam
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
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        print(self.model.summary())

    # TODO: 入力データの型を決める
    def fit(self, data: any, label: any) -> None:
        """
        学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        return self.model.fit(data, label, batch_size=1, epochs=5)

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

class NN_SL(object):
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
        self.model.add(Dropout(0.25))
        self.model.add(Dense(300, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(action_size, activation='softmax'))
        # self.model.compile(loss=huberloss, optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        print(self.model.summary())

    # TODO: 入力データの型を決める
    def fit(self, data: any, label: any,epoch = 8) -> None:
        """
        学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        return self.model.fit(data, label, batch_size=4, epochs=epoch)

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

class NN_emotion(object):
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
        self.model.add(Dense(100, activation='relu', input_dim=14))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(300, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(action_size, activation='softmax'))
        # self.model.compile(loss=huberloss, optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])
        print(self.model.summary())

    # TODO: 入力データの型を決める
    def fit(self, data: any, label: any) -> None:
        """
        学習を実施する

        :param data: 教師データ
        :param label: 教師ラベル
        """

        self.model.fit(data, label, batch_size=32, epochs= 200, validation_split=0.1)

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
    def fit(self, train_x, train_y) -> None:
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
        mid_units1 = int(trial.suggest_discrete_uniform("mid_units1", 50, 300, 50))
        mid_units2 = int(trial.suggest_discrete_uniform("mid_units2", 50, 300, 50))

        #cyr Ai
        model = Sequential()
        model.add(Dense(mid_units1, activation='relu', input_dim=14))
        model.add(Dense(mid_units2, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        # self.model.compile(loss=huberloss, optimizer='adam', metrics=['accuracy'])
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        print("model compile")
        history = model.fit(train_x_copy, train_y_copy, verbose=0, epochs=16, batch_size=8, validation_split=0.1)

        #セッションのクリア
        K.clear_session()
        del model, mid_units1, mid_units2, train_x_copy, train_y_copy
        

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

class PAgent(object):
    """
    深層学習を用いて行動選択を行うエージェント
    """
    # TODO: モデルの保存や読み込み部分を実装する


    def __init__(self, action_size: int) -> None:
        """
        初期化を実施

        :param action_size: 実施出来るアクションの数
        :param greedy_value: グリーディー法を実施するかどうかの確率
        """
        self.model = NN_SL(action_size)
        self.action_size = action_size

    def get_action(self, data: List[Union[int, float]], observation_space: spaces, episode: int, maxEpisode: int) -> Action:
        """
        現在の状態から最適な行動を求める
        :param data: 現在の状態
        :param observation_space: 画面のサイズなどの情報
        :return: 行動(int)
        """
        action_value = self.model.predict(data)
        # NOTE: 一番評価値が高い行動を選択する(Actionにキャストしておく)
        # NOTE: +1しているのは列挙型が0ではなく1スタートだから
        best_action = np.argmax(action_value)

        return best_action

class EmotionAgent(object):
    """
    深層学習を用いて行動選択を行うエージェント
    """
    # TODO: モデルの保存や読み込み部分を実装する


    def __init__(self, emotion_path:str, n_path:str, h_path:str, a_path:str, s_path:str ) -> None:
        """
        初期化を実施

        :param emotion_path: 感情推定器のモデルのパス
        :param n_path: 行動方策のモデルのパス
        :param h_path: 行動方策のモデルのパス
        :param a_path: 行動方策のモデルのパス
        :param s_path: 行動方策のモデルのパス
        """
        
        action_size = 15
        self.model_e = NN_emotion(3)
        self.model_e.load_model(emotion_path)
        self.model_n = NN_SL(action_size)
        self.model_n.load_model(n_path)
        # self.model_h = NN(21)
        self.model_h = NN_SL(action_size)
        self.model_h.load_model(h_path)
        self.model_a = NN_SL(action_size)
        self.model_a.load_model(a_path)
        self.model_s = NN_SL(action_size)
        self.model_s.load_model(s_path)
        self.action_size = action_size
        self.emotion = 3
        self.emotionLog = [1, 0, 0] * 2
        self.e_inputsLog = [1] * 15

    def get_action(self, data: List[Union[int, float]]) -> Action:
        """
        現在の状態から最適な行動を求める
        :param data: 現在の状態
        :param observation_space: 画面のサイズなどの情報
        :return: 行動(int)
        """

        
        if(self.emotion == 0): #happy
            action_value = self.model_h.predict(data)
        elif(self.emotion == 1): #angry
            action_value = self.model_a.predict(data)
        elif(self.emotion == 2): #sad
            action_value = self.model_s.predict(data)
        else: #neutral
            action_value = self.model_n.predict(data)
        
        best_action = np.argmax(action_value)
        if(self.emotion != 0 or True):
            actList = ["1","2","3","4","4","6","7","8","9", "A","B","C","E","S","T"]
            best_action = actList[best_action]
        return best_action


    def emotion_calc(self, data: any) -> None:
        """
        感情情報を更新する

        
        0:喜び
        1:怒り
        2:悲しみ
        3:ニュートラル

        :param data: ゲーム情報
        """
        eInput = self.data2eInputs(data)
        emotion_val = self.model_e.predict(eInput)
        print(emotion_val)
        val =  np.argmax(emotion_val[0])
        if(emotion_val[0][val] > 0.3):
            self.emotion = np.argmax(emotion_val)
        else:
            self.emotion =3
        
        self.emotionLog[3:].extend(emotion_val)

    def get_emotion(self)-> int:
        return self.emotion

    def data2eInputs(self, data):
        downCount = [0,0]
        #最初と最後だけでいいの    
        if(data[27] == "1.0"):
            downCount[0] += 1
        elif(data[92]== "1.0"):
            downCount[1] += 1

        damage = [0,0]
        damage[0] = self.e_inputsLog[2] - float(data[0])
        damage[1] = self.e_inputsLog[3] - float(data[65])

        hp = [float(data[0]),float(data[65])]
        myOppX = [float(data[2]),float(data[67])]
        time = float(data[-1])
        down = [0.0, 0.0]
        if(data[27] == "1.0"):
            down[0] = 1.0
        elif(data[92] == "1.0"):
            down[1] = 1.0


        val = []
        val.extend(damage)
        val.extend(hp)
        val.extend(myOppX)
        val.extend(down)
        val.extend(self.emotionLog)

        self.e_inputsLog = val

        return np.array(val).reshape([1, 14])