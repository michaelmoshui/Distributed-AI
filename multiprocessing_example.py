from multiprocessing import Pool
def calculate_square(n):
    return n * n
if __name__ == "__main__":
    for i in range(50):
        numbers = [1, 2, 3, 4, 5]
        with Pool(processes=4) as pool:
            result = pool.map(calculate_square, numbers)
        print(result)
