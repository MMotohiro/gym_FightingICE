import tkinter as tk
import sys

GUI_WIDTH = 960
GUI_HIGTH = 300
EMOTION_LIST = ["恐怖", "怒り", "悲しい", "嫌悪", "喜び"]
root = tk.Tk()
root.title("emotion survey")
root.geometry(str(GUI_WIDTH)+"x"+str(GUI_HIGTH))

#ラベル表示
Label = tk.Label(text = "強\n(Strong)",width=8, height = 2, font=("", 20))
Label.place(x=0, y = GUI_HIGTH //6 * 2)
Label = tk.Label(text = "弱\n(Weak)",width=8, height = 2, font=("", 20))
Label.place(x=0, y = GUI_HIGTH //6 * 4)

for i,emo in enumerate(EMOTION_LIST):
    # 強い感情
    Button = tk.Button(text=emo,width=8, height = 2, font=("", 20))
    Button.place(x=GUI_WIDTH // 6 * (i+1) - 20, y = GUI_HIGTH //6 * 4)
    #弱い感情
    Button = tk.Button(text=emo,width=8, height = 2, font=("", 20))
    Button.place(x=GUI_WIDTH // 6 * (i+1) - 20, y = GUI_HIGTH //6 * 2)
    
root.mainloop()