import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_team_data():

   #CSV files into DataFrames
    man_utd_df = pd.read_csv('man_utd_stats.csv', header=[0,1])
    chelsea_df = pd.read_csv('chelsea_stats.csv', header=[0,1])

    man_utd_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in man_utd_df.columns.values]
    chelsea_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in chelsea_df.columns.values]
    
    man_utd_df.columns = [col.replace('Unnamed: 0_level_0_', '').replace('Unnamed: 1_level_0_', '').replace('Unnamed: 2_level_0_', '').replace('Unnamed: 3_level_0_', '').replace('Unnamed: 4_level_0_', '').replace('Unnamed: 33_level_0_', '') for col in man_utd_df.columns]
    chelsea_df.columns = [col.replace('Unnamed: 0_level_0_', '').replace('Unnamed: 1_level_0_', '').replace('Unnamed: 2_level_0_', '').replace('Unnamed: 3_level_0_', '').replace('Unnamed: 4_level_0_', '').replace('Unnamed: 33_level_0_', '') for col in chelsea_df.columns]
    
    print(f"Loaded {len(man_utd_df)} Man Utd players, {len(chelsea_df)} Chelsea players")

    print("\nActual column names:")
    print(man_utd_df.columns.tolist())
    
    return man_utd_df, chelsea_df

def load_premier_league_match_statistics():
    
    man_utd_matches_df = pd.read_csv('man_utd_match_stats.csv')
    chelsea_matches_df = pd.read_csv('chelsea_match_stats.csv')
    
    man_utd_matches_df = man_utd_matches_df[man_utd_matches_df['Comp'] == 'Premier League']
    chelsea_matches_df = chelsea_matches_df[chelsea_matches_df['Comp'] == 'Premier League']
    print("\nDEBUG - After filtering:")
    print(f"Man Utd matches remaining: {len(man_utd_matches_df)}")
    print(f"Chelsea matches remaining: {len(chelsea_matches_df)}")

    numerical_cols = ['GF', 'GA', 'xG', 'xGA']
    for col in numerical_cols:
        man_utd_matches_df[col] = pd.to_numeric(man_utd_matches_df[col], errors='coerce')
        chelsea_matches_df[col] = pd.to_numeric(chelsea_matches_df[col], errors='coerce')

    man_utd_matches_df.to_csv('man_utd_match_stats_filtered.csv', index=False)
    chelsea_matches_df.to_csv('chelsea_match_stats_filtered.csv', index=False)

    return man_utd_matches_df, chelsea_matches_df    

def average_starting_teams(man_utd_df, chelsea_df):
    
    # Assuming a 4-3-3 formation for both teams

    def starting_xi(df, team_name):
        # Sort by minutes played and select top 11 players
        
        min_col = None
        for col in df.columns:
            if 'Min' in col:
                min_col = col
                break

        starting_xi = [
            [], # Goalkeeper (1 Value)
            [], # Defenders (4 Values)
            [], # Midfielders (3 Values)
            []  # Forwards (3 Values)
        ]

        # Retrieve Goalkeeper
        gk = df[df['Pos'].str.contains('GK', na=False)].nlargest(1, min_col)
        if not gk.empty:
            starting_xi[0].append(gk.iloc[0]['Player'])
        
        # Retrieve Defenders (pure DF only)
        defenders = df[df['Pos'].str.contains('DF', na=False) & 
                    ~df['Pos'].str.contains('MF', na=False)].nlargest(4, min_col)
        for index, player in defenders.iterrows():
            starting_xi[1].append(player['Player'])
        
        # Retrieve Midfielders (MF but not FW)
        midfielders = df[df['Pos'].str.contains('MF', na=False) & 
                        ~df['Pos'].str.contains('FW', na=False)].nlargest(3, min_col)
        for index, player in midfielders.iterrows():
            starting_xi[2].append(player['Player']) 
        
        # Retrieve Forwards (anyone with FW)
        forwards = df[df['Pos'].str.contains('FW', na=False)].nlargest(3, min_col)
        for index, player in forwards.iterrows():
            starting_xi[3].append(player['Player'])
       
        print(f"\n{team_name} Starting XI (4-3-3):")
        print("="*50)
        print(f"GK: {starting_xi[0]}")
        print(f"DF: {starting_xi[1]}")
        print(f"MF: {starting_xi[2]}")
        print(f"FW: {starting_xi[3]}")
        
        return starting_xi
    man_utd_xi = starting_xi(man_utd_df, "Manchester United")
    chelsea_xi = starting_xi(chelsea_df, "Chelsea")

    return man_utd_xi, chelsea_xi

def extract_player_stats(df):
    # Extract relevant attacking statistics with both total and per 90
    player_stats = df[[
        'Player', 
        'Pos', 
        'Playing Time_Min',
        'Performance_Gls', 
        'Per 90 Minutes_Gls',
        'Performance_Ast', 
        'Per 90 Minutes_Ast',
        'Expected_xG', 
        'Per 90 Minutes_xG',
        'Expected_xAG', 
        'Per 90 Minutes_xAG',
        'Performance_G+A',
        'Per 90 Minutes_G+A'
    ]].copy()
    
    # Rename columns
    player_stats.columns = [
        'Player', 
        'Position', 
        'Minutes',
        'Goals_Total', 
        'Goals_Per90',
        'Assists_Total', 
        'Assists_Per90',
        'xG_Total', 
        'xG_Per90',
        'xAG_Total', 
        'xAG_Per90',
        'G+A_Total',
        'G+A_Per90'
    ]
    
    return player_stats

def extract_opponent_stats(df, team_name):
    # Extract opponent statistics from the DataFrame
    opponent_row = df[df['Player'].str.contains('Opponent Total', case=False, na=False)]

    opponent_stats = {
        'Goals_Conceded': opponent_row['Performance_Gls'].values[0],
        'Goals_Conceded_Per90': opponent_row['Per 90 Minutes_Gls'].values[0],
        'Assists_Conceded': opponent_row['Performance_Ast'].values[0],
        'Assists_Conceded_Per90': opponent_row['Per 90 Minutes_Ast'].values[0],
        'xG_Conceded': opponent_row['Expected_xG'].values[0],
        'xG_Conceded_Per90': opponent_row['Per 90 Minutes_xG'].values[0],
        'xAG_Conceded': opponent_row['Expected_xAG'].values[0],
        'xAG_Conceded_Per90': opponent_row['Per 90 Minutes_xAG'].values[0],
        'G+A_Conceded': opponent_row['Performance_G+A'].values[0],
        'G+A_Conceded_Per90': opponent_row['Per 90 Minutes_G+A'].values[0]
    }

    opponent_stats_series = pd.Series(opponent_stats)

    return opponent_stats_series
    

def starting_xi_averages(player_stats_df, starting_xi, team_df, team_name):
    # Calculate averages for the starting XI
    starting_players = []
    for position in starting_xi:
        starting_players.extend(position)

    starting_xi_df = player_stats_df[player_stats_df['Player'].isin(starting_players)]

    totsl_cols = [
        'Goals_Total',
        'Assists_Total',
        'xG_Total',
        'xAG_Total',
        'G+A_Total'    
    ]

    per90_cols = [
        'Goals_Per90',
        'Assists_Per90',
        'xG_Per90',
        'xAG_Per90',
        'G+A_Per90'    
    ]

    xi_totals = starting_xi_df[totsl_cols].sum()
    xi_per90 = starting_xi_df[per90_cols].sum()

    opponent_stats = extract_opponent_stats(team_df, team_name)

    xi_stats = pd.concat([xi_totals, xi_per90, opponent_stats])

    print(f"\n{team_name} Starting XI Averages:")
    print("="*50)
    print(xi_stats)

    return xi_stats, starting_xi_df

def top3_players_by_metric(man_utd_xi_df, chelsea_xi_df, metric):
    # Generate a table comparing the top 3 players from each team based on the specified metric
    man_utd_top3 = man_utd_xi_df.nlargest(3, metric)[['Player', metric]]
    chelsea_top3 = chelsea_xi_df.nlargest(3, metric)[['Player', metric]]

    table_data = []
    for i in range(3):
        row = [
            i+1,
            man_utd_top3.iloc[i]['Player'],
            f"{man_utd_top3.iloc[i][metric]:.2f}",
            chelsea_top3.iloc[i]['Player'],
            f"{chelsea_top3.iloc[i][metric]:.2f}"
        ]
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=table_data,
                        colLabels=['Rank', 'Man Utd Player', metric, 'Chelsea Player', metric],
                        cellLoc='center',
                        loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.title(f'Top 3 Players by {metric}', fontsize=14)
    plt.tight_layout()

    return fig

def compare_team_statistics_per90(man_utd_stats, chelsea_stats):

    # Compare team statistics per 90 minutes using a bar chart
    team_metrics = ['Goals_Per90', 'Assists_Per90',  'xG_Per90', 'xAG_Per90', 'G+A_Per90',
                    'Goals_Conceded_Per90', 'Assists_Conceded_Per90', 'xG_Conceded_Per90', 'xAG_Conceded_Per90', 'G+A_Conceded_Per90']
    team_metric_names = ['Goals per 90', 'Assists per 90', 'xG per 90', 'xAG per 90', 'G+A per 90',
                         'Goals Conceded per 90', 'Assists Conceded per 90', 'xG Conceded per 90', 'xAG Conceded per 90', 'G+A Conceded per 90']

    man_utd_values = [man_utd_stats[metric] for metric in team_metrics]
    chelsea_values = [chelsea_stats[metric] for metric in team_metrics]
    
    x = np.arange(len(team_metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, man_utd_values, width, label='Manchester United', color='red')
    bars2 = ax.bar(x + width/2, chelsea_values, width, label='Chelsea', color='blue')

    ax.set_title('Starting XI Statistics Comparison')
    ax.set_ylabel('Values per 90 Minutes')
    ax.set_xlabel('Statistical Categories')
    ax.set_xticks(x)
    ax.set_xticklabels(team_metric_names, rotation=45, ha='right', fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

# PREDICTING MANCHESTER UNITED VS CHELSEA

def average_goals_scored_and_conceded(man_utd_matches_df, chelsea_matches_df):

    # Calculate average goals scored and conceded along with xG and xGA for both teams
    man_utd_avg_xG = man_utd_matches_df['xG'].mean()
    man_utd_avg_xGA = man_utd_matches_df['xGA'].mean()
    man_utd_avg_GF = man_utd_matches_df['GF'].mean()
    man_utd_avg_GA = man_utd_matches_df['GA'].mean()

    chelsea_avg_xG = chelsea_matches_df['xG'].mean()
    chelsea_avg_xGA = chelsea_matches_df['xGA'].mean()
    chelsea_avg_GF = chelsea_matches_df['GF'].mean()
    chelsea_avg_GA = chelsea_matches_df['GA'].mean()

    man_utd_averages = {
        'avg_xG':   man_utd_avg_xG,
        'avg_xGA':  man_utd_avg_xGA,
        'avg_GF':   man_utd_avg_GF,
        'avg_GA':   man_utd_avg_GA
    }

    chelsea_averages = {
        'avg_xG':   chelsea_avg_xG,
        'avg_xGA':  chelsea_avg_xGA,
        'avg_GF':   chelsea_avg_GF,
        'avg_GA':   chelsea_avg_GA
    }

    return man_utd_averages, chelsea_averages

def predict_match_score(man_utd_averages, chelsea_averages):

    # Predict match score using adjusted xG model
    man_utd_expected_goals = (man_utd_averages['avg_xG'] + chelsea_averages['avg_xGA']) / 2
    chelsea_expected_goals = (chelsea_averages['avg_xG'] + man_utd_averages['avg_xGA']) / 2

    man_utd_scoring_prob = man_utd_averages['avg_GF'] / man_utd_averages['avg_xG']
    man_utd_conceding_prob = man_utd_averages['avg_GA'] / man_utd_averages['avg_xGA']

    chelsea_scoring_prob = chelsea_averages['avg_GF'] / chelsea_averages['avg_xG']
    chelsea_conceding_prob = chelsea_averages['avg_GA'] / chelsea_averages['avg_xGA']

    adjusted_man_utd_goals = man_utd_expected_goals * man_utd_scoring_prob * chelsea_conceding_prob
    adjusted_chelsea_goals = chelsea_expected_goals * chelsea_scoring_prob * man_utd_conceding_prob
    
    man_utd_predicted = round(adjusted_man_utd_goals)
    chelsea_predicted = round(adjusted_chelsea_goals)

    print("\nPredicted Match Score:")
    print("="*50)
    print(f"Manchester United: {man_utd_predicted} ({adjusted_man_utd_goals}) - Chelsea: {chelsea_predicted} ({adjusted_chelsea_goals})")

    return man_utd_predicted, chelsea_predicted

def validate_xg_prediction(matches_df, team_name, metric_expected, metric_actual, metric_name):
    # Validate xG predictions against actual goals scored/conceded
    expected = matches_df[metric_expected]
    actual = matches_df[metric_actual]

    correlation = expected.corr(actual)

    if correlation >= 0.9:
        validity = "Very Strong Correlation"
    elif correlation >= 0.7:
        validity = "Strong Correlation"
    elif correlation >= 0.5:
        validity = "Moderate Correlation"
    elif correlation >= 0.3:
        validity = "Weak Correlation"
    else:
        validity = "Very Weak Correlation"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(expected, actual, alpha=0.6)

    #Line of best fit
    z = np.polyfit(expected, actual, 1)
    p = np.poly1d(z)
    ax.plot(expected, p(expected), "r--", label='Best Fit Line')

    #45 Degree Line representing perfect correlation
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'g-', label='Perfect Correlation')

    #Labels and Title
    ax.set_xlabel(f'Expected {metric_name} ({metric_expected})')
    ax.set_ylabel(f'Actual {metric_name} ({metric_actual})')
    ax.set_title(f'{team_name}: {metric_name} Validation\nCorrelation: r = {correlation:.3f} ({validity})')

    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    return fig, correlation

def predict_goal_scorers(starting_xi_df, team_name, predicted_goals, correlation_factor):
    # Predict likely goal scorers based on Goals per 90 and xG per 90
    # Adjust weights based on correlation factor
    if correlation_factor >= 0.5:
        weight_actual = 0.7
        weight_xg = 0.3
    elif correlation_factor >= 0.3:
        weight_actual = 0.5
        weight_xg = 0.5
    else:
        weight_actual = 0.3
        weight_xg = 0.7

    starting_xi_df = starting_xi_df.copy()
    starting_xi_df['Scoring_Probability'] = (
        (starting_xi_df['Goals_Per90'] * weight_actual) + 
        (starting_xi_df['xG_Per90'] * weight_xg)
    )
    starting_xi_df = starting_xi_df.sort_values(by='Scoring_Probability', ascending=False)

    if predicted_goals > 0:
        for i in range(int(predicted_goals)):
            scorer = starting_xi_df.iloc[i]
            prob_pct = scorer['Scoring_Probability'] * 100
            print(f"Predicted Goal Scorer {i+1}: {scorer['Player']} (Prob: {prob_pct:.2f}%)")
    else:
        print("No goals predicted for this team.")

    return starting_xi_df

def predicted_match_report(man_utd_xi, chelsea_xi,
                               man_utd_predicted_df, chelsea_predicted_df,
                               man_utd_xi_df, chelsea_xi_df):
    # Generate a visual report of the predicted match outcome
    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.5])

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.text(0.5, 0.5, 'Manchester United vs Chelsea - Match Prediction Report',
                fontsize=20, fontweight='bold', ha='center', va='center')
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')

    man_utd_lineup = []
    chelsea_lineup = []

    # GK
    man_utd_lineup.append(f"{man_utd_xi[0][0]} (GK)")
    chelsea_lineup.append(f"{chelsea_xi[0][0]} (GK)")

    # Defenders
    for df in man_utd_xi[1]:
        man_utd_lineup.append(f"{df} (DF)")
    for df in chelsea_xi[1]:
        chelsea_lineup.append(f"{df} (DF)")

    # Midfielders
    for mf in man_utd_xi[2]:
        man_utd_lineup.append(f"{mf} (MF)")
    for mf in chelsea_xi[2]:
        chelsea_lineup.append(f"{mf} (MF)")

    # Forwards
    for fw in man_utd_xi[3]:
        man_utd_lineup.append(f"{fw} (FW)")
    for fw in chelsea_xi[3]:
        chelsea_lineup.append(f"{fw} (FW)")

    man_utd_scorers = []
    for i in range(int(man_utd_predicted_df)):
        scorer = man_utd_xi_df.iloc[i]
        prob = scorer['Scoring_Probability'] * 100
        man_utd_scorers.append(f"{scorer['Player']} ({prob:.1f}%)")
    
    chelsea_scorers = []
    for i in range(int(chelsea_predicted_df)):
        scorer = chelsea_xi_df.iloc[i]
        prob = scorer['Scoring_Probability'] * 100
        chelsea_scorers.append(f"{scorer['Player']} ({prob:.1f}%)")

    # Build table data
    table_data = []
    
    table_data.append(['Score', str(man_utd_predicted_df), str(chelsea_predicted_df)])
    
    table_data.append(['#', 'Manchester United', 'Chelsea'])

    for i in range(11):
        table_data.append([str(i+1), man_utd_lineup[i], chelsea_lineup[i]])

    table_data.append(['Predicted\nScorers', man_utd_scorers, chelsea_scorers])

    table = ax_table.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.45, 0.45]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    return fig

if __name__ == "__main__":


    man_utd_df, chelsea_df = load_team_data()
    man_utd_xi, chelsea_xi = average_starting_teams(man_utd_df, chelsea_df)

    man_utd_stats = extract_player_stats(man_utd_df)
    chelsea_stats = extract_player_stats(chelsea_df)

    man_utd_xi_averages, man_utd_xi_df = starting_xi_averages(man_utd_stats, man_utd_xi, man_utd_df, "Manchester United") 
    chelsea_xi_averages, chelsea_xi_df = starting_xi_averages(chelsea_stats, chelsea_xi, chelsea_df, "Chelsea")

    per90_metrics = ['Goals_Per90', 'Assists_Per90', 'xG_Per90', 'xAG_Per90', 'G+A_Per90']
    
    for metric in per90_metrics:
        top3_players_by_metric(man_utd_xi_df, chelsea_xi_df, metric)

    compared_stats = pd.DataFrame({
        'Manchester United': man_utd_xi_averages,
        'Chelsea': chelsea_xi_averages
    })  

    print("\nComparison of Starting XI Averages:")
    print("="*50)
    print(compared_stats)

    # Predicting Manchester United vs Chelsea match outcome
    man_utd_matches_df, chelsea_matches_df = load_premier_league_match_statistics()
    man_utd_averages, chelsea_averages = average_goals_scored_and_conceded(man_utd_matches_df, chelsea_matches_df)
    man_utd_predicted_df, chelsea_predicted_df = predict_match_score(man_utd_averages, chelsea_averages)

    # Manchester United xG vs Goals Scored validation
    fig1, corr1 = validate_xg_prediction(man_utd_matches_df, 'Manchester United', 'xG', 'GF', 'Goals Scored')
    # Manchester United xGA vs Goals Conceded validation
    fig2, corr2 = validate_xg_prediction(man_utd_matches_df, 'Manchester United', 'xGA', 'GA', 'Goals Conceded')
    # Chelsea xG vs Goals Scored validation
    fig3, corr3 = validate_xg_prediction(chelsea_matches_df, 'Chelsea', 'xG', 'GF', 'Goals Scored')
    # Chelsea xGA vs Goals Conceded validation
    fig4, corr4 = validate_xg_prediction(chelsea_matches_df, 'Chelsea', 'xGA', 'GA', 'Goals Conceded')

    # Predict likely goal scorers
    man_utd_scorers_df = predict_goal_scorers(man_utd_xi_df, "Manchester United", man_utd_predicted_df, corr1)
    chelsea_scorers_df = predict_goal_scorers(chelsea_xi_df, "Chelsea", chelsea_predicted_df, corr3)

    fig_report = predicted_match_report(
        man_utd_xi, chelsea_xi,
        man_utd_predicted_df, chelsea_predicted_df,
        man_utd_scorers_df, chelsea_scorers_df
    )

    fig = compare_team_statistics_per90(man_utd_xi_averages, chelsea_xi_averages)
    plt.show()




