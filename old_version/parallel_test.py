from time import sleep
from random import random
from multiprocessing import Pool
 
# task to execute in another process
def task(arg):
    # generate a value between 0 and 1
    value = random()
    # block for a fraction of a second to simulate work
    sleep(value)
    # return the generated value
    return arg
 
# entry point for the program
if __name__ == '__main__':
    # create the process pool
    with Pool(processes=4) as pool:
        # call the same function with different data in parallel
        for result in pool.imap(task, range(50)):
            # report the value to show progress
            print(result)