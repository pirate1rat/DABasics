import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([[1.5, 6.7, 3.0], [4.8, 6.2, 1.1]], dtype='float')
#np.astype('int32') - zmienia typ tablicy

print(a, "\n", b)
#print(b.ndim, b.dtype, b.size, b.shape) #jakies podstawowe cechy
#print("total size: ", b.nbytes)

#wyzerowana tablica
tab0 = np.zeros((5, 3), dtype='int32')
#print(tab0)

#jedynkowa tablica
tab1 = np.ones((3, 3, 3))
#print(tab1)

#inna liczba
tab = np.full((2, 2), 65, dtype='b') #np.full_like(tab, 2137)
#print(tab)

#losowe liczby
r = np.random.rand(3, 3)  #tutaj nie tupla
#print(r)
r = np.random.randint(-10, 10, size=(3, 3))
#print(r)

print(np.identity(4))