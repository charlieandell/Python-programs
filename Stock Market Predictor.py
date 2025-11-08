#imports
import yfinance as yf
import pandas as pd
import os
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

#using yfinance to get sp500 price data
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

#fix the warning with utc=True
sp500.index = pd.to_datetime(sp500.index, utc=True)

#debug: check what data we have
print("Data range:", sp500.index.min(), "to", sp500.index.max())

#deleting columns I consider to be irrelevant sp500 price
del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

#limiting data analysed to be after 1990
if sp500.index.min() <= pd.to_datetime("1990-01-01", utc=True):
    sp500 = sp500.loc["1990-01-01":].copy()
    print(f"Filtered data from 1990, shape: {sp500.shape}")
else:
    print(f"Using all available data from {sp500.index.min()}, shape: {sp500.shape}")

# Add technical indicators
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    
    # Get probability predictions
    prob_predictions = model.predict_proba(test[predictors])[:,1]
    
    # PRICE PREDICTION (standard threshold 0.5)
    price_pred = (prob_predictions >= 0.5).astype(int)
    
    # TRADING RECOMMENDATION (different thresholds)
    trading_rec = np.where(prob_predictions >= 0.55, "BUY", 
                          np.where(prob_predictions <= 0.40, "SELL", "HOLD"))
    # Conservative recommendations for selling, as DOWN days are often caused by completely unpredictable events
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Target': test["Target"],
        'Price_Prediction': price_pred,
        'Trading_Recommendation': trading_rec,
        'Confidence': prob_predictions
    }, index=test.index)
    
    return results

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

# Run backtest
predictions = backtest(sp500, model, new_predictors)

# Calculate metrics
accuracy = accuracy_score(predictions["Target"], predictions["Price_Prediction"])

# NEW: Calculate accuracy for BUY recommendations
buy_predictions = predictions[predictions["Trading_Recommendation"] == "BUY"]
if len(buy_predictions) > 0:
    buy_accuracy = accuracy_score(buy_predictions["Target"], buy_predictions["Price_Prediction"])
    buy_count = len(buy_predictions)
else:
    buy_accuracy = 0
    buy_count = 0

# NEW: Calculate accuracy for SELL recommendations
sell_predictions = predictions[predictions["Trading_Recommendation"] == "SELL"]
if len(sell_predictions) > 0:
    # For SELL recommendations, we want the price to actually go DOWN (Target = 0)
    sell_accuracy = accuracy_score(sell_predictions["Target"] == 0, sell_predictions["Price_Prediction"] == 0)
    sell_count = len(sell_predictions)
else:
    sell_accuracy = 0
    sell_count = 0

# NEW: Calculate accuracy for HOLD recommendations (for completeness)
hold_predictions = predictions[predictions["Trading_Recommendation"] == "HOLD"]
if len(hold_predictions) > 0:
    hold_count = len(hold_predictions)
    # Hold accuracy is trickier to define, so we'll just show the count
else:
    hold_count = 0

print("\n=== Final Backtest Performance ===")
print(f"Total periods analyzed: {len(predictions)}")
print(f"\n--- PRICE PREDICTIONS ---")
print(f"UP predictions: {sum(predictions['Price_Prediction'] == 1)}")
print(f"DOWN predictions: {sum(predictions['Price_Prediction'] == 0)}")
print(f"Price Prediction Accuracy: {accuracy:.2%}")

print(f"\n--- TRADING RECOMMENDATION PERFORMANCE ---")
print(f"BUY Recommendations: {buy_count}")
print(f"  → BUY Accuracy: {buy_accuracy:.2%} (how often price actually went UP after BUY signal)")
print(f"SELL Recommendations: {sell_count}")
print(f"  → SELL Accuracy: {sell_accuracy:.2%} (how often price actually went DOWN after SELL signal)")
print(f"HOLD Recommendations: {hold_count}")
print(f"  → (No trade executed)")

print(f"\n--- TRADING STRATEGY SUMMARY ---")
print(f"Total BUY reccomendations: {buy_count}")
print(f"Total SELL reccomendations: {sell_count}")
print(f"Win rate (profitable trades): {((buy_accuracy * buy_count) + (sell_accuracy * sell_count)) / (buy_count + sell_count) if (buy_count + sell_count) > 0 else 0:.2%}")

def predict_tomorrow_only(model, data, predictors):
    """Predict both price direction and trading action for tomorrow"""
    # Get the latest complete data
    latest_data = data[predictors].dropna().iloc[-1:]
    
    # Get probability prediction
    prediction_proba = model.predict_proba(latest_data)[0, 1]
    
    # Price prediction (will it go up/down?)
    price_prediction = 1 if prediction_proba >= 0.5 else 0
    
    # Trading recommendation (should you buy/sell/hold?)
    if prediction_proba >= 0.6:
        trading_action = "BUY"
        action_reason = "High confidence in price increase"
        expected_accuracy = buy_accuracy if buy_count > 0 else "N/A"
    elif prediction_proba <= 0.4:
        trading_action = "SELL" 
        action_reason = "High confidence in price decrease"
        expected_accuracy = sell_accuracy if sell_count > 0 else "N/A"
    else:
        trading_action = "HOLD"
        action_reason = "Uncertain - probability too close to 50%"
        expected_accuracy = "N/A"
    
    tomorrow_date = data.index[-1] + pd.Timedelta(days=1)
    
    result = {
        'Date': tomorrow_date,
        'Price_Prediction': price_prediction,
        'Price_Direction': 'UP' if price_prediction == 1 else 'DOWN',
        'Trading_Action': trading_action,
        'Action_Reason': action_reason,
        'Confidence': prediction_proba,
        'Expected_Accuracy': expected_accuracy
    }
    
    print(f"\n=== TOMORROW'S PREDICTION ===")
    print(f"Date: {tomorrow_date.strftime('%Y-%m-%d')}")
    print(f"Price Prediction: {result['Price_Direction']}")
    print(f"Trading Action: {result['Trading_Action']}")
    print(f"Reason: {result['Action_Reason']}")
    prediction_proba = 1-prediction_proba
    print(f"Confidence Level: {prediction_proba:.2%}")
    if expected_accuracy != "N/A":
        print(f"Historical Accuracy for this signal: {expected_accuracy:.2%}")
    
    return result

# Get tomorrow's prediction
print(f"\n--- Model Verification ---")
print(f"Model trained with {len(new_predictors)} features")
tomorrow_pred = predict_tomorrow_only(model, sp500, new_predictors)
