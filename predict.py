import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

stock = pd.read_csv("sphist.csv")
stock.head()
print(stock.info())

stock['Date'] = pd.to_datetime(stock['Date'])

stock['Date'] > datetime(year=2015, month=4, day=1)
#to tell whether each item in the Data column is after 2015-04-01

stock_sorted = stock.sort_values('Date', ascending=True)
#sort the stock dataframe on the column 'Date'

print(stock_sorted.head(3))

#Pick 3 indicators to compute, and generate a different column for each one. I think some of key indicators will be average price from past 5 days, 365 days and standard deviation over past 5 days and 365 days. I will do this for Volume and Adj Close column.

stock_sorted['avg_price_5'] = stock_sorted['Close'].rolling(window=5).mean().shift(1)

stock_sorted['avg_price_365'] = stock_sorted['Close'].rolling(window=365 ).mean().shift(1)

stock_sorted["ratio_5vs365"] =stock_sorted['avg_price_5'] / stock_sorted['avg_price_365']

stock_sorted["std_price_5"] = stock_sorted['Close'].rolling(window=5).std().shift(1)

stock_sorted["std_price_365"] = stock_sorted['Close'].rolling(window=365).std().shift(1)

stock_sorted['avg_volume_5'] = stock_sorted['Volume'].rolling(window=5).mean().shift(1)

stock_sorted['avg_volume_365'] = stock_sorted['Volume'].rolling(window=365 ).mean().shift(1)

stock_sorted["std_volume_5"] = stock_sorted['Volume'].rolling(window=5).std().shift(1)

stock_sorted["std_volume_365"] = stock_sorted['Volume'].rolling(window=365).std().shift(1)  

stock_sorted['Date_1'] = pd.to_datetime(stock_sorted['Date'])
stock_sorted["last_year"] = stock_sorted["Date_1"].dt.year - 1
year_low = stock_sorted.loc[stock_sorted.groupby('last_year')['Close'].idxmin()]
year_low = year_low[['last_year', 'Close']]
year_low.rename(columns={'Close':'last_low'}, inplace=True)
stock_sorted = stock_sorted.merge(year_low, on = 'last_year')
stock_sorted['last_low_current'] = stock_sorted['last_low'] / stock_sorted['Close']
print(stock_sorted['last_low'])
#pick the ratio - lowst price of last year vs current price as another indicators
#leverage groupby(), and merge() to add column of 'last-low'


print(stock_sorted[stock_sorted['avg_price_365'].isnull()])
#realized there are way fewer trading days per year than calendar year, so for indicator rolling 365 days, the first non-NaN value will appear until June 1951

stock_sorted = stock_sorted[stock_sorted['Date']>datetime(year=1951, month=1, day=3)]
#remove any row from the DateFrame that fall before 1951-01-03

stock_sorted = stock_sorted.dropna(axis=0)
#Use the dropna method to remove any rows with NaN values. 

train = stock_sorted[stock_sorted['Date']<datetime(year=2013, month=1, day=1)]
test =  stock_sorted[stock_sorted['Date']>=datetime(year=2013, month=1, day=1)]
#Generate two new dataframes to use in making our algorithm. train should contain any rows in the data with a date less than 2013-01-01. test should contain any rows with a date greater than or equal to 2013-01-01.

print(train.head())

lr = LinearRegression()
features=['avg_price_365','avg_price_5','last_low_current','avg_volume_365','avg_volume_5','std_price_365','std_price_5','std_price_365','std_volume_365','std_volume_5']

lr.fit(train[features], train['Close'])
predictions = lr.predict(test[features])
MAE = mean_absolute_error(test['Close'], predictions)

print('MAE is {} based on LinearRegression'.format(MAE))

#the error is about 16, will try to add couple more indicators
#Add 2 additional indicators to your dataframe, and see if the error is reduced. You'll need to insert these indicators at the same point where you insert the others, before you clean out rows with NaN values and split the dataframe into train and `test.


#try random forest algorithm to see whether we can improve the predictions
rf = RandomForestRegressor(n_estimators=100, random_state=1)

rf.fit(train[features], train['Close'])
predictions_rf = rf.predict(test[features])
MAE_rf = mean_absolute_error(test['Close'], predictions_rf)

print('MAE is {} based on RandomForestRegressor'.format(MAE_rf))
#the MAE is 349, not sure why this is way higher than LinearRegre








