#####################################PLASKI_POTENCJAL

import matplotlib.pyplot as plt
import numpy as np

# Dane

x = np.array([5, 10, 15, 20, 26, 31, 36, 41, 46], dtype=float)
v1 = np.array([1.87, 2.58, 3.3, 4.04, 4.77, 5.49, 6.15, 6.91, 7.69], dtype=float)
v2 = np.array([0.97, 1.94, 2.91, 3.88, 5.05, 6.02, 6.99, 7.96, 8.93], dtype=float)

# Niepewności
x_err = 0.5      # ±1 mm
y_err = 0.15    # ±0.3 V/m

# Wykres punktów v1 z niepewnościami
plt.errorbar(
    x, v1,
    xerr=x_err, yerr=y_err,
    fmt='o',
    color='blue',
    markersize=2,
    ecolor='blue', capsize=2,
    label='V doświadczalne (z niepewnościami)'
)

#  Regresja LINIowa dopasowana do v2
coef = np.polyfit(x, v2, 1)   # 1 = funkcja liniowa
poly = np.poly1d(coef)

# Linia teoretyczna
x_line = np.linspace(min(x), max(x), 300)
plt.plot(x_line, poly(x_line), color='red', label='V teoretyczne')

# Opisy osi, siatka, legenda
plt.xlabel('Położenie między okładkami [mm]')
plt.ylabel('Potencjał pola [V]')
plt.grid(True, which='both', alpha=0.4)
plt.legend()

plt.show()









############################PLASKI_NATEZENIE
import matplotlib.pyplot as plt
import numpy as np

# Dane

x = np.array([7.5, 12.5, 17.5, 23, 28.5, 33.5, 38.5, 43.5], dtype=float)
v1 = np.array([130, 152, 140, 118.33, 140, 134, 152, 144], dtype=float)
v2 = np.array([194.12, 194.12, 194.12, 194.12, 194.12, 194.12, 194.12, 194.12], dtype=float)

# Niepewności
x_err = 0      # ±1 mm
y_err = 2    # ±0.3 V/m

# Wykres punktów v1 z niepewnościami
plt.errorbar(
    x, v1,
    xerr=x_err, yerr=y_err,
    fmt='o',
    color='blue',
    markersize=2,
    ecolor='blue', capsize=2,
    label='E doświadczalne (z niepewnościami)'
)

#  Regresja LINIowa dopasowana do v2
coef = np.polyfit(x, v2, 1)   # 1 = funkcja liniowa
poly = np.poly1d(coef)

# Linia teoretyczna
x_line = np.linspace(min(x), max(x), 300)
plt.plot(x_line, poly(x_line), color='red', label='E teoretyczne')

# Opisy osi, siatka, legenda
plt.xlabel('Położenie między okładkami [mm]')
plt.ylabel('Natężenie pola [V/m]')
plt.grid(True, which='both', alpha=0.4)
plt.legend()
plt.ylim(0, None)

plt.show()



#############################CYLINDRYCZNY_POTENCJAL
import matplotlib.pyplot as plt
import numpy as np

# Dane

x = np.array([27, 34, 42, 49, 57, 64, 71, 79, 86], dtype=float)
v1 = np.array([7.8, 6.24, 5.1, 4.05, 3.26, 2.47, 1.83, 1.24, 0.7], dtype=float)
v2 = np.array([8.05, 6.64, 5.34, 4.39, 3.46, 2.75, 2.11, 1.45, 0.93], dtype=float)

# Niepewności
x_err = 0.5      # ±1 mm
y_err = 0.15    # ±0.3 V/m

# Wykres punktów v1 z niepewnościami
plt.errorbar(
    x, v1,
    xerr=x_err, yerr=y_err,
    fmt='o',
    color='blue',
    markersize=2,
    ecolor='blue', capsize=2,
    label='V doświadczalne (z niepewnościami)'
)

# Regresja kwadratowa dopasowana do v2
coef = np.polyfit(x, v2, 2)
poly = np.poly1d(coef)

# Linia teoretyczna
x_line = np.linspace(min(x), max(x), 300)
plt.plot(x_line, poly(x_line), color='red', label='V teoretyczne')

# Opisy osi, siatka, legenda
plt.xlabel('Odległość od środka [mm]')
plt.ylabel('Potencjał pola [V]')
plt.grid(True, which='both', alpha=0.4)
plt.legend()

plt.show()




############################CYLINDRYCZNY_NATEZENIE##########
import matplotlib.pyplot as plt
import numpy as np

# Dane
x = np.array([30.5, 38, 45.5, 53, 60.5, 67.5, 75, 82.5])
v1 = np.array([214.29, 135, 134.29, 100, 110, 98.57, 78.75, 70])
v2 = np.array([201.68, 161.87, 135.19, 116.06, 101.67, 91.13, 82.02, 74.56])

# Niepewności
x_err = 0      # ±1 mm
y_err = 2 

# Wykres punktów v1 z niepewnościami
plt.errorbar(
    x, v1,
    xerr=x_err, yerr=y_err,
    fmt='o',
    color='blue',
    markersize=2,
    ecolor='blue', capsize=2,
    label='E doświadczalne (z niepewnościami)'
)

# Regresja kwadratowa dopasowana do v2
coef = np.polyfit(x, v2, 2)
poly = np.poly1d(coef)

# Linia teoretyczna
x_line = np.linspace(min(x), max(x), 300)
plt.plot(x_line, poly(x_line), color='red', label='E teoretyczne')

# Opisy osi, siatka, legenda
plt.xlabel('Odległość od środka [mm]')
plt.ylabel('Natężenie pola [V/m]')
plt.grid(True, which='both', alpha=0.4)
plt.legend()

plt.show()