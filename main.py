

from fastapi import FastAPI
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import datetime as dt
import pickle
from joblib import load
model_saved = load('nba_model.joblib')
app = FastAPI()
def predict_games(team_home, team_away):
    gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable = '10/28/2014', league_id_nullable = '00')
    games = gamefinder.get_data_frames()[0]
    #games = games[['TEAM_NAME','GAME_ID','GAME_DATE', 'MATCHUP', 'WL','FG_PCT','REB','AST','STL','PLUS_MINUS']]
    games = games[['TEAM_NAME','GAME_ID','GAME_DATE', 'MATCHUP', 'WL','FG_PCT','FT_PCT','REB','AST','STL','BLK','TOV','FG3_PCT','PLUS_MINUS']]
    #games = games[games['TEAM_NAME'].str.contains('New York Knicks')]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE']) #convert the game_date column values into a date time object data type
    msk_home = (games['TEAM_NAME'] == team_home) #filter the games dataframe for only the specified home team
    games_30_home = games[msk_home].sort_values('GAME_DATE').tail(30) #get the last 30 games of the specified home team
    home_plus_minus = games_30_home['PLUS_MINUS'].mean() #get the average plus_minus of the specified home team from their last 30 games
    home_ft_pct = games_30_home['FT_PCT'].mean() #get the average ft_pct of the specified home team from their last 30 games
    home_reb = games_30_home['REB'].mean() #get the average number of rebounds for the specified home team from their last 30 games
    home_ast = games_30_home['AST'].mean() #get the average number of assists for the specified home team from their last 30 games
    home_stl = games_30_home['STL'].mean() #get the average number of steals for the specified home team
    home_blk = games_30_home['BLK'].mean() #get the average number of blocks for the specified home team
    home_tov = games_30_home['TOV'].mean() #get the average number of tovs for the specified home team
    home_fg3_pct = games_30_home['FG3_PCT'].mean() #get the average number of 3 point pct for the home team
    home_fg_pct = games_30_home['FG_PCT'].mean() # get the average field goal pct for the home team
    msk_away = (games['TEAM_NAME'] == team_away) #filter the games dataframe for only the specified away team
    games_30_away = games[msk_away].sort_values('GAME_DATE').tail(30) #get the last 30 games of the specified away team
    away_plus_minus = games_30_away['PLUS_MINUS'].mean() #get the average plus_minus of the specified away team from their last 30 games
    away_ft_pct = games_30_away['FT_PCT'].mean() #get the average ft_pct of the specified away team from their last 30 games
    away_reb = games_30_away['REB'].mean() #get the average number of rebounds for the specified away team from their last 30 games
    away_ast = games_30_away['AST'].mean() #get the average number of assists for the specified away team from their last 30 games
    away_stl = games_30_away['STL'].mean() #get the average number of steals for the specified away team
    away_blk = games_30_away['BLK'].mean() #get the average number of blocks for the specified away team
    away_tov = games_30_away['TOV'].mean() #get the average number of tovs for the specified away team
    away_fg3_pct = games_30_away['FG3_PCT'].mean() #get the average number of 3 point pct for the away team
    away_fg_pct = games_30_away['FG_PCT'].mean() # get the average field goal pct for the home team"""

    """Calulating the difference between the home team features and away team features [plus_minus 
    ft_pct, fg_pct, reb, ast, stl, blk tov, fg3_pct """
    AVG_PLUS_MINUS_DIFF = home_plus_minus - away_plus_minus #reflects the performance in both the home and away team
    AVG_FT_PCT_DIFF = home_ft_pct - away_ft_pct
    AVG_FG_PCT_DIFF = home_fg_pct - away_fg_pct
    AVG_REB_DIFF = home_reb - away_reb
    AVG_AST_DIFF = home_ast - away_ast
    AVG_STL_DIFF = home_stl - away_stl
    AVG_BLK_DIFF = home_blk - away_blk
    AVG_TOV_DIFF = home_tov - away_tov
    AVG_FG3_PCT_DIFF = home_fg3_pct - away_fg3_pct
    """create a df containing the quantitative diff in features of the inputted teams so 
    that we can input it into our saved model to make a prediction on which team is going 
    to win"""
    data = {'AVG_PLUS_MINUS_DIFF': [AVG_PLUS_MINUS_DIFF],  'AVG_FG_PCT_DIFF':[AVG_FG_PCT_DIFF], 'AVG_FT_PCT_DIFF': [AVG_FT_PCT_DIFF], 'AVG_FG3_PCT_DIFF': [AVG_FG3_PCT_DIFF], 'AVG_REB_DIFF': [AVG_REB_DIFF], 'AVG_AST_DIFF': [AVG_AST_DIFF], 'AVG_STL_DIFF': [AVG_STL_DIFF], 'AVG_BLK_DIFF': [AVG_BLK_DIFF], 'AVG_TOV_DIFF': [AVG_TOV_DIFF]}
    inputted_teams_df = pd.DataFrame(data)
    predict_home_win = model_saved.predict(inputted_teams_df)[0]
    predict_winning_probability = model_saved.predict_proba(inputted_teams_df)[0][1]
    return { 'result': int(predict_home_win), 'win_probability': float(predict_winning_probability)}

@app.get("/predict_NBA_home_win/")
def predict_games_result(team_home, team_away):
    return predict_games(team_home, team_away)


