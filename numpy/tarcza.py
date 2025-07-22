import numpy as np


n = int(input())

tab = np.zeros((n, n))
for i in range(0, n//2 + 1):
    tab[i : n-i , i : n-i ] = i*2 + 1
print(tab)