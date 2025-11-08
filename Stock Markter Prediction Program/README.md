This python program analyses data imported using yfinance about the S&P500 stock prices,
then uses machine learning to train a model that well predicts whether the price will go up or down,
and also produces a confidence level which is used to detirmine whether the user
should either buy, hold or short the stock depending on how confident the algorithm is in its prediction.
The program has a 56.36% BUY prediction acurracy, and a 54.47% SELL prediction accuracy,
and has been geared to be reasonably cautious, and when making a reccomendation, fairly bullish.
42% of the time, the algorithm reccomended buying, 55% of the time reccomended holding,
and just 3% of the time reccomended selling. The reason for this is that the stock market generally
goes up over time, so a more bullish model achieves higher accuracy percentages, and also because
on days when the S&P500 falls, there has often been some completely unpredictable event
that no algorithm can forsee. Testing the program to make sell recommendations with lower confidence
intervals proved to greatly decrease its accuracy, in som cases below 50%

The margins are very respectable, with a 56% win rate, although the algorithm obviously has no understanding
of why these market movements are happening, and using the program's reccomendations to trade would
likely not be a wise idea.

Nevertheless, it was an interesting project to work on, and taught me a lot about using machine learning
in python, and I will continue to refine this program, and perhaps I shall monitor over the next few months
how accurate its predictions really are.



Here is a typical program output, taken at 22:52 8th November 2025:

      Data range: 1927-12-30 05:00:00+00:00 to 2025-11-07 05:00:00+00:00
      Filtered data from 1990, shape: (9031, 7)
      
      === Final Backtest Performance ===
      Total periods analyzed: 5531
      
      --- PRICE PREDICTIONS ---
      UP predictions: 3901
      DOWN predictions: 1630
      Price Prediction Accuracy: 53.15%
      
      --- TRADING RECOMMENDATION PERFORMANCE ---
      BUY Recommendations: 2342
        → BUY Accuracy: 56.36% (how often price actually went UP after BUY signal)
      SELL Recommendations: 123
        → SELL Accuracy: 54.47% (how often price actually went DOWN after SELL signal)
      HOLD Recommendations: 3066
        → (No trade executed)
      
      --- TRADING STRATEGY SUMMARY ---
      Total BUY reccomendations: 2342
      Total SELL reccomendations: 123
      Win rate (profitable trades): 56.27%
      
      --- Model Verification ---
      Model trained with 10 features
      
      === TOMORROW'S PREDICTION ===
      Date: 2025-11-08
      Price Prediction: DOWN
      Trading Action: HOLD
      Reason: Uncertain - probability too close to 50%
      Confidence Level: 54.86%
      
