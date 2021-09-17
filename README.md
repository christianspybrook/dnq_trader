Deep Q-Learning Stock Trader
============================

Algorithms, Frameworks, and Libraries Demonstrated:
---------------------------------------------------

1. Yahoo API
2. Numpy
3. PyTorch
4. Multilayer Perceptron
5. Deep Reinforcement Learning
6. Argparse
7. Fintech

Project Scope
-------------

This project builds a Deep Reinforement stock trading Agent with a base investment of $20,000. Apple Corporation is used as the stock for the model's demostraation. The closing stock prices are colleccted from the [Yahoo Finance](https://finance.yahoo.com/) website for the years 2013-2018. The Agent is trained on the first half of the dataset. Then, the model is tested on the second half of the data. A randomized trader is run on the same test data to obtain a baseline for final portfolio comparison. The results are plotted for performance analysis.

Stage 1 - Data Sourcing
-----------------------

The data are collected using the [Yahoo_fin](http://theautomatic.net/yahoo_fin-documentation/) library, which makes a call the Yahoo Finance API for stock data.

Stage 2 - Deep Q-Learning Agent Training
----------------------------------------

A Reinforcement Learning Agent is allowed to make trades, based on its prediction of the stock's closing value. The reward is to maximize the portfolio value, which has an initial investment value of $20,000. The Actions allowed are to buy, hold, or sell the stock. Although the demonstration uses only one stock, the model can accept multilpe stocks for trading to represent a full stock portfolio. The Agent is trained for 2,000 episodes. The prediction model is built as a Multilayer Perceptron using PyTorch.

Stage 3 - Performance Test and Random Trades
--------------------------------------------

After training on the first half of the dataset, the model's exploration rate is minimized. This allows the Agent to make trades based on the parameters reached at the end of training. The model makes trades on the second half of the dataset. Then, the exploration rate is adjusted to have the model choose random trading Actions. As the ground truth price of the stock has an increaing trend, this allows the gereation of a baseline to compare model performance.

Stage 4 - Baseline Comparison
-----------------------------

The average portfolio values are nearly the same between the trained model and its randomized version. The trained model's portfolio grows by $3,894, while $3,660 is added through random Actions. The Deep Network shows more utility in constraining the resulting values, as it's portfolios bottom out at $15,231, while the random model's portfolio reaches a low point of $12,540. The results are similar for the maximum portfolio values. The trained Agent's portfolio reaches up to $35,122, while the random model's portfolio hits $44,107, at one point.

Future Work and Model Tuning
----------------------------

1. Include more stocks
3. Optimize hyperparameters (exploration rate, DNN architecture, loss function)
4. Increase number of training episodes
