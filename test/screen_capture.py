import pyautogui
import time
import win32gui
import win32api
from PIL import ImageGrab
from PyQt5.QtWidgets import QApplication


def screen_PIL():
    """使用PIL中的ImageGrab模块获取屏幕"""
    start = time.time()
    img = ImageGrab.grab()
    print('花费时间；', time.time() - start)
    print(img)
    img.show()


# 调用windows API，速度快
def screen_wim32api():
    win32gui.
    win32api.


# 程序会打印窗口的hwnd和title，有了title就可以进行截图了。
def print_hwnd():
    hwnd_title = dict()

    def get_all_hwnd(hwnd, mouse):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
            hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})

    win32gui.EnumWindows(get_all_hwnd, 0)

    for h, t in hwnd_title.items():
        print(h, t)


# PyQt比调用windows API简单很多，而且有windows API的很多优势，比如速度快，可以指定获取的窗口，即使窗口被遮挡。
# 需注意的是，窗口最小化时无法获取截图。
# 首先需要获取窗口的句柄。
def screen_QT():
    hwnd = win32gui.FindWindow(None, '神经网络练习.nb * - Wolfram Mathematica 13.0')
    screen = QApplication.primaryScreen()
    img = screen.grabWindow(hwnd).toImage()
    img.save("screenshot.jpg")


# pyautogui是比较简单的，但是不能指定获取程序的窗口，因此窗口也不能遮挡，不过可以指定截屏的位置，0.04s一张截图，比PyQt稍慢一点，但也很快了。


def screen_pyautogui():
    img = pyautogui.screenshot()  # x,y,w,h
    img.show()
    # img.save('screenshot.png')


if __name__ == '__main__':
    # screen_PIL()
    # print_hwnd()
    # screen_QT()
    screen_pyautogui()