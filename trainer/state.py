from enum import IntEnum, auto

class State(IntEnum):
    """ エージェントの状態 """
    STAND = auto()
    CROUCH = auto()
    AIR = auto()
    DOWN = auto()
