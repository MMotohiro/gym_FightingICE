import numpy as np
import PySimpleGUI as sg

# sg.theme_previewer()
sg.theme('DarkTeal7')

def main():
    ltSurvery()

def ltSurvery():
    layout = []

    layout.append([sg.Text('FICE survey')])
    temp_id = 0

    # layout.append([sg.Text(str(i+1)) for i in range(5)])
    layout.append([sg.Text("予測通り") ,sg.Radio("",group_id = temp_id),sg.Radio("",group_id =  temp_id),sg.Radio("",group_id =  temp_id,default = True),sg.Radio("",group_id =  temp_id),sg.Radio("",group_id =  temp_id),sg.Text("　　驚き")])
    temp_id += 1
    layout.append([sg.Text("　　喜び") ,sg.Radio("",group_id = temp_id),sg.Radio("",group_id =  temp_id),sg.Radio("",group_id =  temp_id,default = True),sg.Radio("", group_id =  temp_id),sg.Radio("", group_id =  temp_id),sg.Text("　悲しみ")])
    temp_id += 1
    layout.append([sg.Text("　　信頼") ,sg.Radio("",group_id = temp_id),sg.Radio("", group_id =  temp_id),sg.Radio("", group_id =  temp_id,default = True),sg.Radio("",group_id =  temp_id),sg.Radio("",group_id =  temp_id),sg.Text("　　嫌悪")])
    temp_id += 1
    layout.append([sg.Text("　　恐れ") ,sg.Radio("",group_id = temp_id),sg.Radio("", group_id =  temp_id),sg.Radio("", group_id = temp_id,default = True),sg.Radio("",group_id =  temp_id),sg.Radio("",group_id =  temp_id),sg.Text("　　怒り")])
    temp_id += 1

    layout.append( [sg.Submit(key='-submit-')])
    window = sg.Window("sample",layout)

    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == 'キャンセル':
            break
        elif event == '-submit-':
            window.close()
            return list(values.values())

    window.close()

if __name__ == "__main__":
    main()