import requests
from time import sleep

url = "http://172.22.38.181:1880/44131030"

while True:
            
    with open("./data/44131030_codo.txt", "r") as f:
        l = f.readline()
        data = l.split(":")[1]

        dataToSend = {}
        dataToSend["valor"] = data.split(",")[0]
        dataToSend["type"] = "max"
        try:
            print("sending max")
            requests.post(url, data = dataToSend, timeout=1)
        except Exception as e:
            print(e)
        sleep(1)
        dataToSend["valor"] = data.split(",")[1]
        dataToSend["type"] = "min"
        try:
            print("sending min")
            requests.post(url, data = dataToSend, timeout=0.5)
        except Exception as e:
            print(e)

    #print the response text (the content of the requested file):

    sleep(600)