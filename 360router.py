import requests
from Crypto.Cipher import AES


def login(password='qwer1234'):
    login_url = 'http://192.168.0.1/router/web_login.cgi'
    login_data = {
        'user': 'admin',
        'pass': get_aes_string(password),
        'from': 1,
    }
    print('loginData:', login_data)

    headers = {
        # 'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'http://192.168.0.1/login_pc.htm',
    }

    resp = requests.post(login_url, data=login_data, headers=headers)
    print('resp headers', resp.headers)
    #  'set-cookie': ['Qihoo_360_login=f27bae15077a77aa7284f83aa4b917ab;path=/'],
    #  connection: 'close',
    # 'content-type': 'text/plain; charset=UTF-8'

    data = resp.json()
    print('resp data:', data)
    if data['success'] == "1" and data["token_id"] != "":
        # success: '1',
        # cookie: 'f27bae15077a77aa7284f83aa4b917ab',
        # token_id: 'aabfabcd42319adab083a064fc50f09d'
        cookie = resp.headers['Set-Cookie']
        token_id = data['token_id']

    return cookie, token_id


def get_aes_string(text):
    # {"rand_key": "2da7df7fa4037e626f2860afae12bd8b91343a062cf030072f5001cf58071bc8"}
    url = 'http://192.168.0.1/router/get_rand_key.cgi?noneed=noneed'
    resp = requests.get(url)
    print('rand_key response', resp.text)
    data = resp.json()

    key_index = data['rand_key'][0:32]
    rand_key = data['rand_key'][32:64]

    key = bytes().fromhex(rand_key)
    iv = '360luyou@install'.encode()
    print('key:', key)
    print('iv:', iv)

    bs = AES.block_size
    pad = lambda s: s + (bs - len(s) % bs) * chr(bs - len(s) % bs)  # PKS7

    cipher = AES.new(key, AES.MODE_CBC, iv)
    cipher_text = cipher.encrypt(pad(text).encode())

    print('cipher_text:', cipher_text.hex())

    return key_index + cipher_text.hex()


def test():
    text = 'qwer1234'
    rand_key = 'dee32b21bf0d967d57da6e6d5bc1d3d0e374c968d7143438d3103b5cb20eebfe'

    # key: e374c968d7143438d3103b5cb20eebfe
    # iv: 3336306c75796f7540696e7374616c6c
    # cipher_text: 1fe84fd2f3c8cf4bafd6d2151180a5f5
    key_index = rand_key[0:32]
    rand_key = rand_key[32:64]

    # key,iv 都为16字节
    key = bytes().fromhex(rand_key)
    iv = '360luyou@install'.encode()
    print('key:', key)
    print('iv:', iv)

    bs = AES.block_size
    pad = lambda s: s + (bs - len(s) % bs) * chr(bs - len(s) % bs)  # PKS7

    cipher = AES.new(key, AES.MODE_CBC, iv)
    cipher_text = cipher.encrypt(pad(text).encode())

    print('cipher_text:', cipher_text.hex())
    return key_index + cipher_text.hex()


if __name__ == '__main__':
    # r = requests.get('https://www.98zudisw.xyz/')
    # print(r.json())

    # test()
    cookie, token_id = login()
    print(cookie, token_id)
