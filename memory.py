import numpy as np
from collections import deque
from typing import List

class Memory:
    """ experience replayを行う際に過去の行動情報を保持するクラス """
    def __init__(self, max_size=1000) -> None:
        """
        初期化を実施

        :param max_size: 過去の情報をどれだけ保持するか
        """
        self.buffer = deque(maxlen=max_size)


    # TODO: experienceの型を定義する
    def get_last_data(self) -> any:
        """
        最後に保存した情報を返す

        :return: 最後に保存した情報
        """
        return self.buffer[-1]

    # TODO: experienceの型を定義する
    def add(self, experience: any) -> None:
        """
        過去の情報を追加する

        :param experience: 情報
        """
        self.buffer.append(experience)

    # TODO: experienceの型を定義する
    def sample(self, batch_size: int) -> List[any]:
        """
        保持している情報からランダムに複数取得する

        :param batch_size: 取得する情報数
        """
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self) -> int:
        """
        現在保持している情報数を返す

        :return: 情報数
        """
        return len(self.buffer)