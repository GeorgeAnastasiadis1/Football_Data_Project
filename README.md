# Football_Data_Project

**visualisations.py** loads the Manchester United and Chelsea CSVs, flattens/cleans the columns, and picks a 4-3-3 starting XI for each side based on minutes by position (GK, four defenders, three midfielders, three forwards). It then builds per-90 attacking metrics (e.g., Goals, Assists, xG, xGA, G+A), aggregates team/lineup averages, and reads match-by-match data to filter Premier League fixtures and retrieve key stats (GF, GA, xG, xGA). With those inputs, it generates comparison bar charts and correlation plots, prints the chosen lineups and summary tables, and produces a lightweight xG-blended match prediction (including likely scorers) before displaying the figures with Matplotlib.

**Helpful Football Terminology**
xG = expected goals (very popular football statistic used currently)
xAG = expected assists
xGA = expected goals conceded
GF = goals for the team
GA = goals against the team
G+A = goals and assists

**Prediction Methodology**
The score prediction is based on an expected goals approach. Each team's expected goals in the match are calculated by averaging their attacking strength (average xG) with their opponents' defensive weakness (average xGA),  then adjusted using performance coefficients that measure finishing ability (actual goals / xG) and defensive performance (actual goals conceded / xGA) to account for how clinical or wasteful each team is at converting chances.
Methodology is validated by calculating correlation coefficients between xG and actual goals for both attacking and defensive metrics. 

Goalscorer prediction used a weighted probability that combined each player in starting XIs actual average goals and average xG, with weights adjusted based on the attacking xG correlation. A stronger correlation meant actual goals were weighted more heavily and vice versa. The top N players by scoring probability are predicted to score, where N equals the predicted team goals, resulting in a final prediction of Manchester United 1-2 Chelsea with specific goal scorers identified.

For **team_statistics.py**, I was able to webscrape for one of the tables but had to copy and paste for the match statistics table. I would have used an API aswell but all of these were quite expensive and had no free trials. 
