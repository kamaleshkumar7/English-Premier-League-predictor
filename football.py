import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf

epl_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1819/E0.csv")
epl_1617 = epl_1617[['HomeTeam','AwayTeam','FTHG','FTAG']]
epl_1617 = epl_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
epl_1617.head()
team = ['Arsenal','Burnley','Leicester','Cardiff','Man United','Bournemouth','Wolves','Fulham','Huddersfield','Watford','Newcastle','Liverpool','Southampton','Crystal Palace','Chelsea','Tottenham','Brighton','Everton','Man City','West Ham']
print(team)
print("Enter team name as given above")

goal_model_data = pd.concat([epl_1617[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           epl_1617[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
poisson_model.summary()

#number of goals each team scores
hteam = input("Enter home team ") #home team 
ateam = input("Enter away team ") #away team

#home team goals
hteamgoals = poisson_model.predict(pd.DataFrame(data={'team': hteam, 'opponent': ateam,
                                       'home':1},index=[1]))
print(hteamgoals)

#away team goals
ateamgoals = poisson_model.predict(pd.DataFrame(data={'team': ateam, 'opponent': hteam,
                                       'home':0},index=[1]))
print(ateamgoals)

#match probabillity
def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
prob = simulate_match(poisson_model, hteam, ateam, max_goals=10)
win = np.sum(np.tril(prob, -1))
draw = np.sum(np.diag(prob))
loss = np.sum(np.triu(prob, 1))
print("home team win percentage")
print(win*100)
print("Away team win percentage")
print(loss*100)
print("Draw percentage")
print(draw*100)
