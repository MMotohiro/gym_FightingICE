import sys, os, datetime
import msvcrt
from time import sleep
from ltSurvey import ltSurvery

sys.path .append('../')
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field


#LIST OF EMOTION
KEY_LIST = {'1':"h0", '4':"h1", '7':"h2", '2':"a0", '5':"a1", '8':"a2", '3':"s0", '6':"s1", '9':"s2",}

gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242), callback_server_parameters=CallbackServerParameters());
manager = gateway.entry_point
args = sys.argv

dt_now = datetime.datetime.now()

REPLAY_NAME = "HPMode_KeyBoard_KeyBoard_2021.08.03-17.42.19"
# get replay path
try:
    REPLAY_NAME = args[1]
except:
    print("replay file is not found")
    pass

LOG_NAME = REPLAY_NAME + ".csv"
LOG_PATH = "./logs/" + LOG_NAME
LT_PATH =  "./logs/lt_" + LOG_NAME
#init replay
print("Replay: Loading")
replay = manager.loadReplay(REPLAY_NAME) # Load replay data

print("Replay: Init")
replay.init()
e_list = []
lt_list = []
r_temp = 1

sleep(5)

# Main process
for i in range(12000): # Simulate 100 frames
    # print("Replay: Run frame", i)    
    framedata = replay.getFrameData()
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key != b'\xe0':
            key = key.decode()
        else:
            key = msvcrt.getch()

        if key in KEY_LIST:
            e_list.append(((framedata.getRound(),framedata.getFramesNumber()),KEY_LIST[key]))
            print(KEY_LIST[key])
        elif key == 'q':
            break


    sys.stdout.flush()
    replay.updateState()
    # lt servey
    r_num = framedata.getRound()
    # if r_temp != r_num and r_num >= 2:
    #     r_temp = r_num
    #     lt_temp = ltSurvery()
    #     lt_list.append(lt_temp)
        

    if replay.getState().name() == "CLOSE":
        break
    
# end replay
print("Replay: Close")
print(e_list)

#dataset with dataframe




txt_list = []
for i in e_list:
    txt_list.append(",".join(map(str, i)))

with open(LOG_PATH, mode='w') as f:
    f.writelines(txt_list)

# with open(LT_PATH, mode='w') as f:
#     f.writelines(lt_list)

replay.close()

sys.stdout.flush()

gateway.close_callback_server()
gateway.close()


# def flatten(data ,emotion = None) -> np.ndarray:
#         """
#         NNに入力できるように配列に変形する

#         :param data: 変形したいデータ
#         :return: 変形後のデータ
        
#         TODO: 感情情報にも対応させる
#         """

#         result = np.zeros((1, len(data)+1 ))
#         result[0][-1] = emotion


#         for i in range(len(data)):
#             result[0][i] = data[i]
        


#         return result