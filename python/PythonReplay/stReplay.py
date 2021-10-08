import sys, os, datetime
import msvcrt
from time import sleep
sys.path .append('../')
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field

KEY_LIST = {b'H':"happy", b'K':"angry", b'P':"sad", b'M':"surprise"}

gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242), callback_server_parameters=CallbackServerParameters());
manager = gateway.entry_point

dt_now = datetime.datetime.now()

REPLAY_NAME = "HPMode_KeyBoard_KeyBoard_2021.08.03-17.42.19"
LOG_NAME = dt_now.strftime('%Y%m%d%H%M%S')+".txt"
LOG_PATH = "./logs/" + LOG_NAME

print("Replay: Loading")
replay = manager.loadReplay(REPLAY_NAME) # Load replay data

print("Replay: Init")
replay.init()
e_list =[]

# Main process
for i in range(12000): # Simulate 100 frames
    # print("Replay: Run frame", i)    
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key != b'\xe0':
            key = key.decode()
        else:
            key = msvcrt.getch()

        if key in KEY_LIST:
            framedata = replay.getFrameData()
            e_list.append(((framedata.getRound(),framedata.getFramesNumber()),KEY_LIST[key]))
            print(KEY_LIST[key])
        elif key == 'q':
            break


    sys.stdout.flush()

    replay.updateState()

    if replay.getState().name() == "CLOSE":
        break
    

print("Replay: Close")
print(e_list)
txt_list = []
for i in e_list:
    txt_list.append("".join(map(str, i)))

with open(LOG_PATH, mode='w') as f:
    f.writelines(e_list)

replay.close()

sys.stdout.flush()

gateway.close_callback_server()
gateway.close()