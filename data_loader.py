from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
import pandas as pd

# source data (Apple stock closing price 2013-2018)
df = get_data('AAPL', '2013-01-01', '2018-01-01')['close'].to_frame().rename(
			  columns={'close': 'AAPL'})

# save data
df.to_csv('/data/apple_data.csv', index=False)

print(df)

plt.title('Apple Stock: 2013-2018')

plt.plot(df)

plt.xlabel('Year')
plt.ylabel('Close')

plt.savefig('/apple_stock.png')

plt.show()