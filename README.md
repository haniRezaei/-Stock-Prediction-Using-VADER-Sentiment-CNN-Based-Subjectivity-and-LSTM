# A Comparison of VADER and Subjectivity-Enhanced VADER


This project explores the use of financial news sentiment to predict stock market movements and prices. We focus on comparing VADER sentiment analysis with a subjectivity-aware variant that incorporates a CNN-based SubjObj_Score, allowing us to distinguish subjective opinions from objective facts.



Financial news strongly influences investor behavior and market dynamics. Traditional sentiment analysis tools like VADER provide scores for positive, negative, and neutral sentiment but cannot differentiate between subjective opinions and factual statements. To address this, we trained a 1D Convolutional Neural Network (CNN) on the Cornell Subjectivity Dataset. The CNN assigns a SubjObj_Score to each headline, which is then combined with VADER sentiment scores and stock market indicators for predicting stock movements and prices.

# Dataset

The study uses the Daily News for Stock Market Prediction dataset (Sun, 2016, Kaggle), which integrates DJIA stock price data with daily headlines from Reddit WorldNews. Stock data include Open, High, Low, Close, Adjusted Close, Volume, and technical indicators such as returns, volatility, RSI, and MACD. Headlines cover the top 25 news items per day, selected based on relevance. The dataset also includes a binary label indicating whether the DJIA moved up or down the next day. Data are chronologically aligned to simulate realistic forecasting conditions.

# Preprocessing and Feature Construction

For preprocessing, the 25 daily headlines are combined into a single text field, cleaned, tokenized, lemmatized, and normalized. Sentiment is automatically extracted using VADER and VADER with CNN-based subjectivity filtering (SubjObj_Score). Daily sentiment scores are aggregated and integrated with stock indicators, technical features, and day-of-week variables to produce a structured, time-ordered dataset suitable for predictive modeling.

# Predictive Modeling

Two main tasks are performed: stock movement classification and stock price regression. For classification, multiple machine learning models are evaluated, including Logistic Regression, Random Forest, XGBoost, and others, using time-series cross-validation. For regression, a Long Short-Term Memory (LSTM) neural network is trained on sequences of 30 previous days to forecast the next-day adjusted closing price. All sentiment pipelines share the same LSTM configuration to ensure fair comparison.

# VADER vs. Subjectivity-Aware VADER

Incorporating SubjObj_Score improves the quality of sentiment features by filtering out objective content. Classification models using subjectivity-aware sentiment achieve higher accuracy in predicting stock movement than VADER alone. Similarly, LSTM-based regression benefits from these enhanced features, producing lower errors and higher R² scores. This demonstrates that accounting for subjectivity in financial news improves both directional and numerical predictions of stock market performance.






