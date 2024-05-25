import MiAI as ma
import numpy as np
import time
from multiprocessing import Pool, shared_memory
from functools import partial

# class ExampleMA(ma.Model):
#     def __init__(self):
#         super().__init__()
#         self.layers = [
#             ma.Dense(784, 1024),
#             ma.ReLU(),
#             ma.Dense(1024, 1024),
#             ma.ReLU(),
#             ma.Dense(1024, 512),
#             ma.ReLU(),
#             ma.Dense(512, 10),
#             ma.Softmax()
#         ]

def f(ind, shm_name, shape, dtype):

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    X_shared = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    res = [X_shared[ind] + 1] * 999999

    existing_shm.close()
    
    return res

if __name__ == "__main__":
    X = np.array([1, 3, 5, 7, 9, 11])

    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
    X_shared = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
    np.copyto(X_shared, X)

    with Pool() as pool:
        partial_f = partial(f, shm_name=shm.name, shape=X.shape, dtype=X.dtype)
        all_results = pool.map(partial_f, range(6))
        
    shm.close()
    shm.unlink()
    print(len(all_results))

