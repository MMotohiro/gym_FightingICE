import json, csv
import glob

DATA_PATH = "./learningData/csv/"

def main():
    print("a")
    action_size = 15
    rawData = None
    files = glob.glob(DATA_PATH +"*.csv")
    datas = []
    sample = [[0]*20 for _ in range(3)] 
   
    # read csv
    for file in files:
        with open(file, 'r') as f:
            try:
                reader = csv.reader(f)
                datas.extend([row for row in reader])
            except:
                pass



    for i, data in enumerate(datas):
        if(data[-1] != "None"): 
            val = int(data[-1])
            try:
                hpDiff = int(float(data[0])*400 - float(data[65])*400)
                if(hpDiff > 50):
                    sample[0][val] += 1
                elif(hpDiff < -50):
                    sample[1][val] += 1
                else:
                    sample[2][val] += 1
            except:
                print(val)
    
    print(sample)

if( __name__ == "__main__"):
    main()
