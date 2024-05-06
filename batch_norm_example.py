import MiAI
import numpy as np

if __name__ == "__main__":
    bn = MiAI.BatchNorm(2)
    # print(bn.gamma)
    # print(bn.beta)
    bn.params['G'] = 1
    bn.params['B'] = 0
    X = np.random.randint(0, 5, size=(3, 2))
    # beta = np.array([[2,3]])
    print(X)
    print()
    print(bn(X))
    # print(beta.shape)
    # print(X * beta)
    # bn(X)
    # print(arr)

    # mean_dim = tuple(j for j in range(arr.ndim) if j != 1) # average across all but the 1st axis
    # mean = np.mean(arr, axis=mean_dim)
    # print(mean)



