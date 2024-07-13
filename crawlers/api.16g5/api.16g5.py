import pandas as pd
import requests



def gain():
    url = "http://api.16g5.com//SQL/vod1.zip"

    resp = requests.get(url)
    with open("vod1.zip", "wb") as file:
        file.write(resp.content)

        df = pd.read_csv(file,sep="|-|")

        print(df)


gain()