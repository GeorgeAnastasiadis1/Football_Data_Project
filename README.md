# Football_Data_Project

**visualisations.py** loads the Manchester United and Chelsea CSVs, flattens/cleans the columns, and picks a 4-3-3 starting XI for each side based on minutes by position (GK, four defenders, three midfielders, three forwards). It then builds per-90 attacking metrics (e.g., Goals, Assists, xG, xAG, G+A), aggregates team/lineup averages, and reads match-by-match data to filter Premier League fixtures and retrieve key stats (GF, GA, xG, xGA). With those inputs, it generates comparison bar charts and correlation plots, prints the chosen lineups and summary tables, and produces a lightweight xG-blended match prediction (including likely scorers) before displaying the figures with Matplotlib.

For **team_statistics.py**, I was able to webscrape for one of the tables but had to copy and paste for the match statistics table. I would have used an API aswell but all of these were quite expensive and had no free trials. 
