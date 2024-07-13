import requests
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


def download_all(url: str, dir: str):

    xml = requests.get(url).text
    # print(xml)
    root = ET.fromstring(xml)
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    print(root.tag)

    # for Contents in root.findall("{ns}Contents" ):
    #     print(Contents.tag)

    for contents in tqdm(root.findall(f"{ns}Contents")):
        key = contents.findtext(f"{ns}Key")
        path = os.path.join(dir, key)
        size = contents.findtext(f"{ns}Size")
        # print(key,size)

        if os.path.isfile(path) and str(os.path.getsize(path)) == size:
            continue
        download(url+key, path)


def download(url: str, path: str):
    """ download file from url """
    # print(f'download : {url} -> {path}')
    content = requests.get(url).content
    with open(path, 'wb+') as f:
        f.write(content)


if __name__ == "__main__":
    download_all('https://api.360zqf.com/oss/zqf-yxf-a/', r'E:\Download\miji')
