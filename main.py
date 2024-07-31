import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and prepare data
data = pd.read_csv('data.csv')

# Reshape data
monthly_data = pd.melt(data, id_vars=['ITEM DESCRIPTION'],
                       value_vars=[col for col in data.columns if 'JANUARY' in col or 'FEBRUARY' in col or 'MARCH' in col or 'APRIL' in col or 'MAY' in col or 'JUNE' in col or 'JULY' in col or 'AUGUST' in col or 'SEPTEMBER' in col or 'OCTOBER' in col or 'NOVEMBER' in col or 'DECEMBER' in col],
                       var_name='Month_Year', value_name='Consumption')

# Extract year and month
monthly_data['Year'] = monthly_data['Month_Year'].str.split('_').str[0]
monthly_data['Month'] = monthly_data['Month_Year'].str.split('_').str[1]

# Correct the reversed columns
monthly_data['Year'], monthly_data['Month'] = monthly_data['Month'], monthly_data['Year']

# Create a date column
monthly_data['Date'] = pd.to_datetime(monthly_data['Month'] + ' ' + monthly_data['Year'], format='%B %Y')

# Drop unnecessary columns
monthly_data = monthly_data.drop(columns=['Month_Year'])

# Sort by date
monthly_data = monthly_data.sort_values(by='Date')

# Set Date as index
monthly_data.set_index('Date', inplace=True)

# Prepare for each item
items = monthly_data['ITEM DESCRIPTION'].unique()

future_predictions_all_items = []

for item in items:
    # Filter data for the current item
    item_data = monthly_data[monthly_data['ITEM DESCRIPTION'] == item].copy()  # Use .copy() to avoid warnings

    # Prepare features and target
    item_data['Month'] = item_data.index.month
    item_data['Year'] = item_data.index.year
    item_data['Day_of_Week'] = item_data.index.dayofweek

    # Example target variable: Predict next month's consumption
    item_data['Next_Consumption'] = item_data['Consumption'].shift(-1)
    item_data = item_data.dropna()

    # Features and target
    X = item_data[['Month', 'Year', 'Day_of_Week']]
    y = item_data['Next_Consumption']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize XGBoost regressor
    model = xgb.XGBRegressor(objective='reg:squarederror')

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Item: {item} - Mean Squared Error: {mse}')

    # Example: Predict next month's consumption for the next 12 months
    future_dates = pd.date_range(start=item_data.index.max() + pd.DateOffset(months=1), periods=12, freq='ME')
    future_features = pd.DataFrame({
        'Month': future_dates.month,
        'Year': future_dates.year,
        'Day_of_Week': future_dates.dayofweek
    })

    # Predict future consumption
    future_predictions = model.predict(future_features)

    # Create DataFrame for future predictions
    future_df = pd.DataFrame({'Date': future_dates, 'Item': item, 'Predicted_Consumption': future_predictions})
    future_predictions_all_items.append(future_df)

# Combine all future predictions into a single DataFrame
final_predictions = pd.concat(future_predictions_all_items)
print(final_predictions)