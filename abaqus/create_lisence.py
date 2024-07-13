def read_bin(path):
    with open(path, 'rb') as f:
        data = f.read()
        print(data)
        print(str(data))

        # 提取License信息
        license_info = data[0x0:0x6]
        print(license_info)
        mac=int.from_bytes(license_info)
        print(mac)
        print(hex(mac))
        
if __name__ == '__main__':
    read_bin(r"D:\Code\C++\GL\src\hardware_info.bin")