from py4j.java_gateway import get_field

class RoleBaseAgent(object):
    """
    ルールベースで行動選択を行うエージェント
    """

    def __init__(self, gateway):
        print("init enemy class")
        self.gateway = gateway

    def close(self):
        pass

    def getInformation(self, frameData, isControl):
        # Load the frame data every time getInformation gets called
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        print(x)
        print(y)
        print(z)

      # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        pass

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.player = player
        self.gameData = gameData
        self.simulator = self.gameData.getSimulator()
        self.isGameJustStarted = True
        print("done initalize")
        return 0

    def input(self):
        # The input is set up to the global variable inputKey
        # which is modified in the processing part
        return self.inputKey


    def processing(self):
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
            print("Start processing")
            self.isGameJustStarted = True
            return

        if not self.isGameJustStarted:
            # Simulate the delay and look ahead 2 frames. The simulator class exists already in FightingICE
            self.frameData = self.simulator.simulate(self.frameData, self.player, None, None, 17)
            #You can pass actions to the simulator by writing as follows:
            #actions = self.gateway.jvm.java.util.ArrayDeque()
            #actions.add(self.gateway.jvm.enumerate.Action.STAND_A)
            #self.frameData = self.simulator.simulate(self.frameData, self.player, actions, actions, 17)
        else:
            # If the game just started, no point on simulating
            self.isGameJustStarted = False

        self.cc.setFrameData(self.frameData, self.player)
        
        distance = self.frameData.getDistanceX()
        
        my = self.frameData.getCharacter(self.player)
        energy = my.getEnergy()
        my_x = my.getX()
        my_state = my.getState()
        
        opp = self.frameData.getCharacter(not self.player)
        opp_x = opp.getX()
        opp_state = opp.getState()
        
        xDifference = my_x - opp_x

        

        if self.cc.getSkillFlag():
          # If there is a previous "command" still in execution, then keep doing it
            self.inputKey = self.cc.getSkillKey()
            return
        # We empty the keys and cancel skill just in case
        self.inputKey.empty()
        self.cc.skillCancel()
        

        # Following is the brain of the reflex agent. It determines distance to the enemy
        # and the energy of our agent and then it performs an action
        if not my_state.equals(self.gateway.jvm.enumerate.State.AIR) and not my_state.equals(self.gateway.jvm.enumerate.State.DOWN):
            # self.cc.commandCall("STAND_A")
            # If not in air
            if distance > 150:
                # If its too far, then jump to get closer fast
                self.cc.commandCall("FOR_JUMP")
            elif energy >= 20:
                # High energy projectile
                self.cc.commandCall("STAND_FB")
            elif (distance > 100) and (energy >= 50):
                # 距離があるとき滑り込み
                self.cc.commandCall("STAND_D_DB_BB")
            elif opp_state.equals(self.gateway.jvm.enumerate.State.AIR): # If enemy on Air
                # Perform a big punch
                self.cc.commandCall("STAND_B")
            elif distance > 100:
                # Perform a quick dash to get closer
                self.cc.commandCall("6 6 6")
            else:
                # Perform a kick in all other cases, introduces randomness
                self.cc.commandCall("STAND_A")
        else:
            # Perform a kick in all other cases, introduces randomness
            self.cc.commandCall("B")


    class Java:
        implements = ["aiinterface.AIInterface"]





    # def get_action(self, data: Dict, observation_space: spaces) -> Action:
    #     """
    #     現在の状態から最適な行動を求める
    #     :param data: 現在の状態
    #     :param observation_space: 画面のサイズなどの情報
    #     :return: 行動(int)
    #     """
    #     distance = abs(data["self"]["X"] - data["opp"]["X"])

    #     if data["opp"]["Energy"] >= 300 and data["self"]["HP"] - data["opp"]["HP"] <= 300:
    #         return Action.FOR_JUMP

    #     elif data["self"]["State"] is not State.AIR and data["self"]["State"] is not State.DOWN:

    #         if distance > 150:
    #             return Action.FOR_JUMP
    #         elif data["self"]["Energy"] >= 50:
    #             return Action.STAND_D_DF_FB
    #         elif distance > 100 and data["self"]["Energy"] >= 50:
    #             return Action.STAND_D_DB_BA
    #         elif data["opp"]["State"] is State.AIR:
    #             return Action.STAND_FB
    #         elif distance > 100:
    #             return Action.DASH
    #         else:
    #             return Action.STAND_B

    #         """
    #         elif  distance <= 150 and (data["self"]["State"] is State.AIR or data["self"]["State"] is State.DOWN)
    #             and ()
    #         """

    #     else:
    #         return Action.STAND_B



