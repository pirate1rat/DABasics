import numpy as np

def ustaw():
    tab = np.array([[1, 2, 3, 4], 
                    [5, 6, 7, 8], 
                    [9, 10, 11, 12]])
    #na dwa sposoby
    print(tab[1][2])
    print(tab[1, 2])

    tab[1][2] = 999
    print(tab[1][-2])

    #kolumny i wiersze, standardowy slicing
    print(tab[2][:]) #tab[2, :]
    print(tab[:][2]) #tab[:, 2]

    #podstawienie
    tab[2][:] = [0, 0, 0, 0]
    print(tab)

    ar1 = np.array([[1, 2, 3]])
    ar1 = np.repeat(ar1, 3, axis=0)
    print(ar1)

    ar1 = ar1.reshape((9, 1))
    print(ar1)

    #kopiowanie
    ar1 = tab.copy()
    print(ar1)

def matma():
    tab1 = np.array([1, 2, 3])
    tab2 = np.array([5, 6, 7])
    print(tab1 * tab2)
    print(tab1 + tab2)
    #wszystko mozna, dzielic, mnozyc, dodawac, tab += 2137, np.sin(), cos itp

    #algebra liniowa
    m1 = np.array([[1, 2], [2, 1]])
    m2 = np.array([[1, 4], [0, 5]])
    print(np.matmul(m1, m2))
    print(np.linalg.det(m1), np.linalg.det(m2))
    m3 = np.vstack([m1, m2])
    print(m3)
    m3 = np.stack([m1, m2])
    print(m3)
    #wszystko co mozna na macierzach


    #statystyka
    print(np.min(m2), np.max(m2, axis=0))
    # np.

def inne():
    tab = np.array([[1, 5, 3, 4], [5, 6, 3, 8], [9, 1, 4, 0]])
    print(tab[tab > 5])
    #mozna indexowac inna tablica
    print(tab[[0, 1]])

def sortowanie():
    ra = np.random.randint(10, size=(3, 4))
    kt_index_po_sortowaniu = np.argsort(ra)
    print(kt_index_po_sortowaniu)

#ustaw()
#matma()
#inne()
sortowanie()