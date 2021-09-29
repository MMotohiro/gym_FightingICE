from action import Action
from state import State
from typing import Dict
from gym import spaces

class RoleBaseAgent(object):
    """
    ルールベースで行動選択を行うエージェント
    """

    def __init__(self):
        pass

    def get_action(self, data: Dict, observation_space: spaces) -> Action:
        """
        現在の状態から最適な行動を求める
        :param data: 現在の状態
        :param observation_space: 画面のサイズなどの情報
        :return: 行動(int)
        """
        distance = abs(data["self"]["X"] - data["opp"]["X"])

        if data["opp"]["Energy"] >= 300 and data["self"]["HP"] - data["opp"]["HP"] <= 300:
            return Action.FOR_JUMP

        elif data["self"]["State"] is not State.AIR and data["self"]["State"] is not State.DOWN:

            if distance > 150:
                return Action.FOR_JUMP
            elif data["self"]["Energy"] >= 300:
                return Action.STAND_D_DF_FC
            elif distance > 100 and data["self"]["Energy"] >= 50:
                return Action.STAND_D_DB_BB
            elif data["opp"]["State"] is State.AIR:
                return Action.STAND_F_D_DFA
            elif distance > 100:
                return Action.DASH
            else:
                return Action.STAND_B

            """
            elif  distance <= 150 and (data["self"]["State"] is State.AIR or data["self"]["State"] is State.DOWN)
                and ()
            """

        else:
            return Action.STAND_B



