import sys, os
import msvcrt
from time import sleep
sys.path .append('../')
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field

KEY_LIST = {'a':"a", 'b':"b",'c':"c"}

gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242), callback_server_parameters=CallbackServerParameters());
manager = gateway.entry_point

print("Replay: Loading")
replay = manager.loadReplay("HPMode_KeyBoard_KeyBoard_2021.08.03-17.42.19") # Load replay data

print("Replay: Init")
replay.init()
e_list =[]

# Main process
for i in range(12000): # Simulate 100 frames
    # print("Replay: Run frame", i)    
    if msvcrt.kbhit():
        key = msvcrt.getch().decode()
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
replay.close()

sys.stdout.flush()

gateway.close_callback_server()
gateway.close()