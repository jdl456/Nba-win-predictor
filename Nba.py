from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
#call class to recieve info on player lebron james and store in player_info variable
#player_info = commonplayerinfo.CommonPlayerInfo(player_id=2544)
#player_info.available_seasons.get_dict()
#players.find_players_by_full_name('james')

gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable = '11/10/2014', league_id_nullable = '00')
games = gamefinder.get_data_frames()[0]
print(games.columns)
#games = games[['TEAM_NAME','GAME_ID','GAME_DATE', 'MATCHUP', 'WL','FG_PCT','REB','AST','STL','PLUS_MINUS']]
games = games[['TEAM_NAME','GAME_ID','GAME_DATE', 'MATCHUP', 'WL','FG_PCT','FT_PCT','REB','AST','STL','BLK','PLUS_MINUS']]
#games = games[games['TEAM_NAME'].str.contains('New York Knicks')]


#games = games[games['PTS'] >= 150]


games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE']) #convert the game_date column values into a date time object data type
games = games.sort_values('GAME_DATE')
games = games.reset_index(drop = True) 
"""groups the df by Team_name and adds a new column that is the rolling average plus/minus rating of the team within 30 games """
#feature one of the model the avg plus minus rating of team
games['AVG_PLUS_MINUS'] = games.groupby('TEAM_NAME')['PLUS_MINUS'].transform(lambda x: x.rolling(35, closed = 'left').mean())
#feature two of the model the avg fg pct of the team
games['AVG_FT_PCT'] = games.groupby('TEAM_NAME')['FT_PCT'].transform(lambda x: x.rolling(35, closed = 'left').mean())
games['AVG_REB'] = games.groupby('TEAM_NAME')['REB'].transform(lambda x: x.rolling(35, closed = 'left').mean())
games['AVG_AST'] = games.groupby('TEAM_NAME')['AST'].transform(lambda x: x.rolling(35, closed = 'left').mean())
games['AVG_STL'] = games.groupby('TEAM_NAME')['STL'].transform(lambda x: x.rolling(35, closed = 'left').mean())
games['AVG_BLK'] = games.groupby('TEAM_NAME')['BLK'].transform(lambda x: x.rolling(35, closed = 'left').mean())
games['AVG_FG_PCT'] = games.groupby('TEAM_NAME')['FG_PCT'].transform(lambda x: x.rolling(35, closed = 'left').mean())
#print(games.tail(30))
#final dataframe: one row for one game
#two columns: 1.result of the game target 2:score stat comparing two teams : feature
#msk only contains data for away games
msk = games['MATCHUP'].str.contains('@')
games_away = games[msk]
games_home = games[~msk]


games_merge = pd.merge(games_away, games_home, on = 'GAME_ID', suffixes = ('_home', '_away'))
"""THe difference between the home and away team in avg_plus_minus will be the feature that helps us determine the outcome of our predictive model"""
games_merge['AVG_PLUS_MINUS_DIFF'] = games_merge['AVG_PLUS_MINUS_home'] - games_merge['AVG_PLUS_MINUS_away'] #reflects the performance in both the home and away team
games_merge['AVG_FT_PCT_DIFF'] = games_merge['AVG_FT_PCT_home'] - games_merge['AVG_FT_PCT_away']
games_merge['AVG_FG_PCT_DIFF'] = games_merge['AVG_FG_PCT_home'] - games_merge['AVG_FG_PCT_away']
games_merge['AVG_REB_DIFF'] = games_merge['AVG_REB_home'] - games_merge['AVG_REB_away']
games_merge['AVG_AST_DIFF'] = games_merge['AVG_AST_home'] - games_merge['AVG_AST_away']
games_merge['AVG_STL_DIFF'] = games_merge['AVG_STL_home'] - games_merge['AVG_STL_away']
games_merge['AVG_BLK_DIFF'] = games_merge['AVG_BLK_home'] - games_merge['AVG_BLK_away']
#print(games_merge)
games_model = games_merge[['WL_home', 'AVG_PLUS_MINUS_DIFF', 'AVG_FG_PCT_DIFF', 'AVG_FT_PCT_DIFF', 'AVG_REB_DIFF','AVG_AST_DIFF','AVG_STL_DIFF','AVG_BLK_DIFF']].dropna()
print(games_model)
games_model['WL_home'] = games_model['WL_home'].map({'W':1, 'L':0})
df_train, df_test = train_test_split(games_model, stratify = games_model['WL_home'], test_size = 0.2, random_state = 7)
target = 'WL_home'
#print(df_train)
#print(df_test)
x_train = df_train.drop(columns = target) #avg_plus_minus_diff training data for model
y_train = df_train[target] #WL_home training data for model

x_test = df_test.drop(columns = target) #avg_plus_minus_diff testing data for model
y_test = df_test[target] #WL_home testing data for model
#hyperparameter - tuning (tweak the pararameters used before the learning process to improve accuracy)
clf = LogisticRegression('l2')
#clf = xgb.XGBClassifier(random_state = 7, use_label_encoder = False)
clf.fit(x_train, y_train)
print(y_test.shape)
y_pred = clf.predict(x_test)
#plot logistic regression curve with black points and red line
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted losses', 'Predicted wins'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual losses', 'Actual wins'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
print(accuracy_score(y_test, y_pred))