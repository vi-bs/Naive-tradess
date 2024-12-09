import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from googletrans import Translator

# Initialize Google Translator
translator = Translator()

# Translate text to a selected language
def translate_text(text, lang='en'):
    try:
        return translator.translate(text, dest=lang).text
    except Exception as e:
        print(f"Error in translation: {e}")
        return text

# Download stock data
def download_stock_data(ticker):
    stock_data = yf.download(ticker, start="2020-01-01", end="2023-12-01")
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data.dropna(inplace=True)
    return stock_data

# Prepare data for prediction
def prepare_data(stock_data):
    stock_data['Target'] = stock_data['Close'].shift(-1)  # Predict next day's closing price
    stock_data.dropna(inplace=True)
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_data['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Fun trading scenarios
def trading_scenarios(predicted_price, actual_price):
    if predicted_price > actual_price:
        return "🔼 The model predicts an upward trend. Buying might be a good option!"
    elif predicted_price < actual_price:
        return "🔽 The model predicts a downward trend. Selling might be safer!"
    else:
        return "🔄 The model predicts no significant movement. Holding could be wise!"

# Quiz feature
def trading_quiz(y_test, y_pred, lang):
    print("\n--- Scenario-Based Quiz ---")
    score = 0

    for i in range(3):  # Show 3 quiz questions
        predicted_price = y_pred[i]
        actual_price = y_test.iloc[i]

        # Display Scenario
        print(f"\nScenario {i + 1}:")
        print(f"Today's Closing Price: {actual_price:.2f}")
        print(f"Predicted Next Day Price: {predicted_price:.2f}")
        print(translate_text("What will you do? [Buy, Sell, Hold]", lang))

        # Taking user input
        print(translate_text("Choose an action:", lang))
        print(f"1. {translate_text('Buy', lang)}")
        print(f"2. {translate_text('Sell', lang)}")
        print(f"3. {translate_text('Hold', lang)}")

        choice = input(f"Your choice (1/2/3): ").strip()

        if choice == "1":
            user_choice = "buy"
        elif choice == "2":
            user_choice = "sell"
        elif choice == "3":
            user_choice = "hold"
        else:
            print(translate_text("Invalid input. Please choose a valid option.", lang))
            continue

        # Logic to compare the user's choice with the correct answer
        suggestion = trading_scenarios(predicted_price, actual_price)
        correct_choice = "buy" if predicted_price > actual_price else "sell" if predicted_price < actual_price else "hold"

        if user_choice == correct_choice:
            print("✅ Correct! " + suggestion)
            score += 1
        else:
            print("❌ Incorrect. " + suggestion)

    print(f"\n{translate_text('Your Quiz Score:', lang)} {score}/3")
    if score == 3:
        print("🏆 Excellent! You're learning the trading basics quickly!")
    elif score == 2:
        print("👍 Good job! A bit more practice will make you better!")
    else:
        print("👨‍💻 Keep practicing! Understanding market trends takes time.")

# Display Graph
def display_graph(stock_data, y_test, y_pred):
    plt.figure(figsize=(10,6))

    # Plot the actual vs predicted closing prices
    plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Closing Prices', color='blue')
    plt.plot(stock_data.index[-len(y_test):], y_pred, label='Predicted Closing Prices', color='red', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function
def main():
    print("Welcome to the Stock Trading Assistant!\n")

    # Language selection for Indian languages
    languages = {
        "1": "hi",  # Hindi
        "2": "bn",  # Bengali
        "3": "ta",  # Tamil
        "4": "te",  # Telugu
        "5": "mr",  # Marathi
        "6": "gu",  # Gujarati
        "7": "kn",  # Kannada
        "8": "ml",  # Malayalam
        "9": "pa",  # Punjabi
        "10": "or", # Odia
        "11": "as", # Assamese
        "12": "ur", # Urdu
        "13": "ne", # Nepali
        "14": "si", # Sinhala
    }

    print("Choose a language:")
    for key, value in languages.items():
        print(f"{key}. {translate_text('Language', value)}")

    lang_choice = input("Choose a language by number (1/2/3/4/5/6/...): ").strip()

    if lang_choice in languages:
        lang = languages[lang_choice]
    else:
        print(translate_text("Invalid choice, defaulting to Hindi.", "hi"))
        lang = "hi"

    # Stock selection
    print(translate_text("Choose a stock to trade with:", lang))
    print("1. Apple (AAPL)")
    print("2. Tesla (TSLA)")
    print("3. Microsoft (MSFT)")

    stock_choice = input("Enter your choice (1/2/3): ").strip()

    if stock_choice == "1":
        ticker = "AAPL"
    elif stock_choice == "2":
        ticker = "TSLA"
    elif stock_choice == "3":
        ticker = "MSFT"
    else:
        print(translate_text("Invalid choice, defaulting to Apple.", lang))
        ticker = "AAPL"

    print(f"Downloading data for {ticker}...")
    stock_data = download_stock_data(ticker)

    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(stock_data)

    print("Training the model...")
    model = train_model(X_train, y_train)

    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Display results
    mse = mean_squared_error(y_test, y_pred)
    print(f"{translate_text('Model Mean Squared Error:', lang)} {mse:.2f}")

    # Start the quiz immediately
    trading_quiz(y_test, y_pred, lang)

    # After quiz, show the graph
    display_graph(stock_data, y_test, y_pred)

if __name__ == "__main__":
    main()
