import os


def flatten(directory: str, is_delete_folder: bool):
    fs = os.listdir(directory)
    for name in fs:
        print(name)
        fullname=os.path.join(directory,name)
        if os.path.isdir(fullname):
            # d
            pass



if __name__ == '__main__':
    directory = 'G:/s'
    flatten(directory, False)
