Stock Trading Assistant

A stock trading assistant tool that helps you predict stock prices using machine learning, and allows users to interact with trading scenarios through quizzes. The tool uses historical stock data to train a Random Forest model to predict future stock prices, and it also includes a multilingual quiz feature for trading practice.

Features

Stock Price Prediction: The model predicts the next day's stock price based on historical data.
Scenario-based Trading Quiz: The tool tests your trading knowledge and decision-making based on predicted stock prices.
Multilingual Support: Choose from various Indian languages to interact with the application.
Visualization: The tool generates graphs comparing actual vs predicted stock prices.
Requirements

To run this tool, you will need Python 3.x and the following libraries:

yfinance for downloading stock data.
numpy for numerical operations.
pandas for data manipulation.
matplotlib for data visualization.
scikit-learn for machine learning algorithms.
googletrans for language translation.
To install the required libraries, you can use the following command:

pip install yfinance numpy pandas matplotlib scikit-learn googletrans==4.0.0-rc1
How It Works

1. Stock Data Download
The tool uses the yfinance library to download historical stock data for selected companies (Apple, Tesla, or Microsoft) from Yahoo Finance. It then calculates the daily return for the stock.

2. Data Preparation
The data is preprocessed to include relevant features for prediction, such as Open, High, Low, Close, and Volume. The model is trained to predict the next day’s closing price.

3. Model Training
A RandomForestRegressor from scikit-learn is used to train the model on the historical stock data. The model learns the relationship between the stock's features and its future price.

4. Prediction & Evaluation
After training, the model makes predictions for the test data, and the mean squared error (MSE) is calculated to evaluate the model's performance.

5. Trading Quiz
The tool presents trading scenarios based on predicted stock prices and asks the user to choose whether they would buy, sell, or hold. The user's choice is compared with the correct decision, and feedback is provided.

6. Graphical Display
A graph is displayed to show the actual vs predicted stock prices over time.

How to Use

Run the Application
Execute the script in your Python environment:
python stock_trading_assistant.py
Choose a Language
Upon running, you will be prompted to choose a language from a list of Indian languages.
Select a Stock
Choose a stock (Apple, Tesla, or Microsoft) for analysis.
Predict Stock Price
The model will download the data, train the model, and make predictions.
Take the Quiz
Answer the trading quiz based on the predicted stock prices.
View Results
After the quiz, the application will show a graph comparing the actual and predicted stock prices.
Example Output

Welcome to the Stock Trading Assistant!

Choose a language:
1. Hindi
2. Bengali
3. Tamil
...
Choose a stock to trade with:
1. Apple (AAPL)
2. Tesla (TSLA)
3. Microsoft (MSFT)

...
Model Mean Squared Error: 3.25
Scenario 1:
Today's Closing Price: 150.23
Predicted Next Day Price: 152.67
What will you do? [Buy, Sell, Hold]

Choose an action:
1. Buy
2. Sell
3. Hold

Your choice (1/2/3): 1
✅ Correct! The model predicts an upward trend. Buying might be a good option!
...
Your Quiz Score: 2/3
Good job! A bit more practice will make you better!
Conclusion

This tool is designed to help users understand stock market trends and practice trading decisions based on machine learning predictions. By interacting with the trading quiz and using the prediction model, users can learn about stock trading in a fun and interactive way.

