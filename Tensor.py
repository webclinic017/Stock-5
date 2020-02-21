import numpy as np
import cProfile
import LB

import time
import threading

# set global variable flag
flag = 1


def get_input():
    global flag
    keystrk = input('Press a key \n')
    # thread doesn't continue until key is pressed

    print('You pressed: ', keystrk)
    flag = False
    print('flag is now:', flag)


def normal():
    while flag == 1:
        print('normal stuff')

        if flag == False:
            print('The while loop is now closing')


i = threading.Thread(target=get_input)

i.start()

normal()
