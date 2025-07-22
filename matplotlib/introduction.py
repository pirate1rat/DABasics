import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_income_data():
    np.random.seed(0)
    month_copies = np.random.randint(20, size=(5, 4))
    months_df = pd.DataFrame(month_copies, index=['jan', 'feb', 'mar', 'apr', 'may'],
                                        columns=['game1', 'game2', 'game3', 'dlc'])
    games_prices = np.array([[3, 6, 10, 2]])
    prices = pd.DataFrame(games_prices, index=['price'], columns=['game1', 'game2', 'game3', 'dlc'])
    income_df = pd.DataFrame(np.dot(prices, months_df.transpose()), index=['income'],
                                                                columns=['jan', 'feb', 'mar', 'apr', 'may'])
    months_df['income'] = income_df.T #income_df.transpose()
    return months_df, income_df

np.random.seed(5)
x = np.sort(np.random.randint(10, size=(6,)))
y = np.random.randint(20, size=(6,))
# x = np.array([1, 3, 5, 8, 10, 11, 15])
# y = np.array([2, 3, 7, 10, 14, 8, 9])
# print(x, "\n", y)

data, in_data = create_income_data()
# fig, axes = plt.subplots(figsize=(6, 6))
# axes.plot(in_data.columns, in_data.loc['income'].values)
# axes.set(title='sprzedaż I półrocze',
#          xlabel='miesiąc',
#          ylabel='dochód w tys. $',)
# plt.show()

#data.hist(bins=8, figsize=(7, 7))

# print(data)
# data.plot(kind='pie', y='game2')
# plt.show()
#fig.savefig('wykres.png')

ts = pd.Series(np.random.randn(1000))
ts = ts.cumsum()
ts.plot()
plt.show()