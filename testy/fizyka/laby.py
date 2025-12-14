laby = [4.0, 4.5, 4.75, 4.5, 4.5, 4.75]
laby_waga = 0.4
wejsc = [2.5, 4, 3.5]
wejsc_waga = 0.6
sr = 0
for i in laby:
    sr += i*laby_waga
for i in wejsc:
    sr += i*wejsc_waga
print(sr/(laby_waga*len(laby)+wejsc_waga*len(wejsc)))