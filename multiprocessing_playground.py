'''
Takeaways:
~ passing data into processes take a long time, try to set up computation within the process
~ Tasks that are already very efficient on one thread will suffer when trying to multiprocess
~ when the number of processes matches the number of jobs, computation is faster
~ more efficient when there is approximately one process per cpu core, not ideal for more processes or number of processes < cpu cores
'''
from multiprocessing import Pool
import numpy as np
import time
import matplotlib.pyplot as plt

def add(inp):
    a, b, m = inp
    for i in range(int(m)):
        a = a + b
    return a

if __name__ == "__main__":

    avgs = []
    n = 1
    m = 640
    while n <= 32:
        
        inputs = [(np.random.randn(200, 1000), np.random.randn(200, 1000), m) for _ in range(n)]
        
        avg = 0
        s = time.time()
        with Pool() as pool:
            results = pool.map(add, inputs)
        e = time.time()
        avg += (e - s)
        print(avg)
        
        avg /= 1
        
        avgs.append([n, avg])
       
        n *= 2
        m /= 2
    
    avgs = np.array(avgs)
    plt.plot(avgs[:, 0], avgs[:, 1])