# Nba-win-predictor
""" THE PURPOSE OF THE PROJECT AND A RUNDOWN ON HOW I IMPLEMENTED IT"""
The purpose of this project is to make a solid statistical prediction on what NBA team would win when matched up against another one. How I implemented this was
by gathering NBA games data through the NBA_Api and filtering the data to contain the 30 game rolling average of features such as the team's average assists, rebounds,
FG_PCT, and other features that would contribute to an NBA team winning. I would then split the dataframe to two dataframes with one being the home team and the other
being the away team. I then merged these two dataframes by their common GAME_ID and added new columns that represented the difference between the home and the away team
statistical features such as 'AVG_PLUS_MINUS_DIFF', 'AVG_FT_PCT_DIFF', 'AVG_REB_DIFF' and more. These would be critical features in determining the target of a win or loss
of the home team. I took these statistical features with the target of a win or loss of the home team and made it a dataframe. This dataframe would be split into a training
dataframe and testing dataframe that I would fit into the logistic regression model. This model included ridge regularization to prevent overfitting the data. Overfitting
is when the model takes in too much noise from the data and when coming across new data to make a prediction on, it would not do well in generalizing and making a solid 
prediction. I then fitted the x_train dataframe and y_train dataframe into my Logistic Regression model to train it. After that I then tested my predictive model to make
a prediction on X_test dataframe with the command y_pred = clf.predict(x_test). The y_pred represents the predictions on whether the home team won or loss in those specific
games. I tested the accuracy score of my predictive model by the command print(accuracy_score(y_test, y_pred)) and out of a sample of 2148 NBA games it predicted 68% 
or 1468 of those games correctly. With all this being done I then defined a function "def predict_games(team_home, team_away): " that would take in a users inputted 
strings of the home team and away team and return whether the home team had won or loss and their probability of winning. This same function would be implemented in the 
main.py file which deployed my predictive model through fastapi.


""" HOW TO RUN THE PREDICTIVE MODEL AND GET A PREDICTION ON A MATCHUP"""
In order to run my predictive model to output a prediction on a matchup would be by first running the NBA.py file to instantiate the predictive model. (done in VSC
through command: py Nba.py) then you would enter the command (uvicorn main:app --reload) in the terminal to launch the API that uses the predictive model. You can then
go to the link http://127.0.0.1:8000/docs and input A home team and away team which would return a 1 representing a win for the home team or a 0 representing a win for 
the away team. This will be followed by the probability of the home team winning.
