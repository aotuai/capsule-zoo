import numpy as np

array = np.reshape(np.arange(0, 5), (1, 1, 5, 1))
boo = [5, 6, 7, 8, 9]

for thin, thing in zip(boo, array.flatten()):
    print(thin, thing)
