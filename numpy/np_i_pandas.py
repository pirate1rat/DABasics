import numpy as np
import pandas as pd
import matplotlib

np.random.seed(0)
month_copies = np.random.randint(20, size=(5, 4))
#print(month_copies)
months_df = pd.DataFrame(month_copies, index=['jan', 'feb', 'mar', 'apr', 'may'],
                                     columns=['game1', 'game2', 'game3', 'dlc'])
games_prices = np.array([3, 6, 10, 2])
games_prices = games_prices.reshape(1, 4)
prices = pd.DataFrame(games_prices, index=['price'], columns=['game1', 'game2', 'game3', 'dlc'])
print(prices)

income_df = pd.DataFrame(np.dot(prices, months_df.transpose()), index=['income'],
                                                             columns=['jan', 'feb', 'mar', 'apr', 'may'])
print(income_df)
months_df['income'] = income_df.T #income_df.transpose()
print(months_df)
