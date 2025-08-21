import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

data=pd.read_csv("C:/Users/Sanghraj/PyCharmMiscProject/player_data_sample.csv")

print(data.head(22))
print(data.info())

assert data.shape[0] == 22, f"Expected 22 players, but found {data.shape[0]}"

valid_roles = {'Batsman', 'Bowler', 'WK', 'Allrounder'}
invalid_roles = set(data['role']) - valid_roles
assert not invalid_roles, f"Invalid roles found: {invalid_roles}"

data['target_count'] = (data['perc_selection'] * 20000).round().astype(int)

def generate_team(data, max_attempts=1000):
    for _ in range(max_attempts):
        sample = data.sample(11, weights=data['perc_selection'], replace=False)
        roles = sample['role'].tolist()
        if {'Batsman', 'Bowler', 'WK', 'Allrounder'}.issubset(roles):
            return tuple(sorted(sample['player_code'].tolist()))
    raise ValueError("Unable to generate a valid team after multiple attempts.")

set_teams = set()
attempts = 0
max_attempts = 1000000
while len(set_teams) < 20000 and attempts < max_attempts:
    team = generate_team(data)
    set_teams.add(team)
    attempts += 1
print(f"Generated {len(set_teams)} unique teams")

all_players = [player for team in set_teams for player in team]
freq_counter = Counter(all_players)
data['teamcount'] = data['player_code'].map(freq_counter).fillna(0).astype(int)

data['actualpercselection'] = data['teamcount'] / len(set_teams)
data['percerror'] = ((data['actualpercselection'] - data['perc_selection']) / data['perc_selection']).round(4)

within_error = data['percerror'].abs() <= 0.05
qualified_players = within_error.sum()

print(f"Players within ±5% error: {qualified_players} out of 22")
print("\nSummary Table:")
print(data[['match_code', 'player_code', 'player_name', 'role', 'team', 'perc_selection', 'teamcount', 'actualpercselection', 'percerror']])



plt.figure(figsize=(10, 5))
plt.bar(data['player_name'], data['percerror'])
plt.axhline(0.05, color='green', linestyle='--', label='+5% error')
plt.axhline(-0.05, color='red', linestyle='--', label='-5% error')
plt.xticks(rotation=90)
plt.title('Player Selection Percentage Error')
plt.legend(loc=0)

plt.show()


summary_of_columns = [
    'match_code', 'player_code', 'player_name', 'role', 'team',
    'perc_selection', 'teamcount', 'actualpercselection', 'percerror'
]
data[summary_of_columns].to_csv('summary_of_accuracy.csv', index=False)


with open('report_error.txt', 'w') as f:
    f.write(f"Players within ±5% error: {qualified_players} out of 22\n\n")

    exceeded = data[~within_error][['player_code', 'player_name', 'percerror']]
    if not exceeded.empty:
        f.write("Players exceeding ±5% error threshold:\n")
        for _, row in exceeded.iterrows():
            f.write(f" - {row['player_name']} (Code: {row['player_code']}, Error: {row['percerror']:.4f})\n")
    else:
        f.write("All players are within the ±5% error threshold.\n")

print("\n Files generated: 'summary_of_accuracy.csv' and 'report_error.txt'")




team_data = []
for team_id, team in enumerate(set_teams):
    for player_code in team:
        player_row = data[data['player_code'] == player_code].iloc[0]
        team_data.append({
            "match_code": player_row['match_code'],
            "player_code": player_code,
            "player_name": player_row['player_name'],
            "role": player_row['role'],
            "team": player_row['team'],
            "perc_selection": player_row['perc_selection'],
            "team_id": team_id
        })

team_df = pd.DataFrame(team_data)
team_df.to_csv("team_df.csv", index=False)


def evaluate_team_accuracy(team_df):
    print(" Evaluating Fantasy Team Accuracy...\n")

    print(f" team_df shape: {team_df.shape}")
    total_teams = team_df['team_id'].nunique()
    total_players = team_df['player_code'].nunique()
    print(f" Total unique teams: {total_teams}")
    print(f" Total unique players: {total_players}")

    role_per_team = team_df.groupby('team_id')['role'].nunique()
    missing_role_teams = role_per_team[role_per_team < 4].count()
    print(f" Teams missing at least one role: {missing_role_teams} / {total_teams}\n")

    player_ref = team_df.drop_duplicates(subset='player_code')[
        ['match_code', 'player_code', 'player_name', 'role', 'team', 'perc_selection']
    ].copy()

    team_counts = team_df.groupby('player_code')['team_id'].nunique().reset_index(name='actual_team_count')
    merged = pd.merge(player_ref, team_counts, on='player_code', how='left')
    merged['actual_team_count'] = merged['actual_team_count'].fillna(0).astype(int)

    merged['expected_team_count'] = (merged['perc_selection'] * total_teams).round(0).astype(int)
    merged['actual_perc_selection'] = merged['actual_team_count'] / total_teams

    merged['perc_error'] = (
        (merged['actual_perc_selection'] - merged['perc_selection']) / merged['perc_selection']
    ).round(4)

    merged['perc_selection'] = (merged['perc_selection'] * 100).round(2)
    merged['actual_perc_selection'] = (merged['actual_perc_selection'] * 100).round(2)
    merged['perc_error'] = (merged['perc_error'] * 100).round(2)

    accuracy_df = merged[[
        'match_code', 'player_code', 'player_name', 'role', 'team',
        'perc_selection', 'expected_team_count', 'actual_team_count',
        'actual_perc_selection', 'perc_error'
    ]].sort_values('player_code')

    within_5 = accuracy_df[accuracy_df['perc_error'].abs() <= 5]
    outside_5 = accuracy_df[accuracy_df['perc_error'].abs() > 5]

    print(" Accuracy KPIs:")
    print(f" Players within ±5% relative error: {within_5.shape[0]} / {accuracy_df.shape[0]}")
    print(f" Players outside ±5% error: {outside_5.shape[0]}")
    print(f" Minimum error: {accuracy_df['perc_error'].min():.2f}%")
    print(f" Maximum error: {accuracy_df['perc_error'].max():.2f}%\n")

    if not outside_5.empty:
        print(" Players with >5% relative error:\n")
        print(outside_5[['player_code', 'player_name', 'perc_selection',
                         'actual_perc_selection', 'perc_error']].to_string(index=False))

    accuracy_df.to_csv("accuracy_summary.csv", index=False)
    print("\n Accuracy summary saved as 'accuracy_summary.csv'")

with open("evaluation_output.txt", "w") as f:
    with redirect_stdout(f):
        evaluate_team_accuracy(team_df)

print(" All files generated: team_df.csv, accuracy_summary.csv, evaluation_output.txt")




