import pandas as pd

# Load CSV files
deliveries = pd.read_csv('data/deliveries.csv')
matches = pd.read_csv('data/matches.csv')

# Add a new column 'inning_id' for grouping
deliveries['inning_id'] = deliveries['match_id'].astype(str) + '-' + deliveries['inning'].astype(str)

# ✅ Filter innings that played at least 19 overs
valid_innings = deliveries.groupby('inning_id')['over'].max()
valid_innings = valid_innings[valid_innings >= 19].index
deliveries = deliveries[deliveries['inning_id'].isin(valid_innings)]

# ✅ Define more frequent checkpoints
checkpoints = list(range(5, 20))  # from over 5 to 19

# Store features in a list
features = []

for checkpoint in checkpoints:
    for inning_id, group in deliveries.groupby('inning_id'):
        group_cp = group[group['over'] < checkpoint]
        if group_cp.empty:
            continue

        total_runs = group_cp['total_runs'].sum()
        wickets = group_cp['player_dismissed'].notnull().sum()
        overs = group_cp['over'].nunique()

        # ✅ This is correct — final target is full innings total
        final_score = group['total_runs'].sum()

        features.append({
            'inning_id': inning_id,
            'runs': total_runs,
            'wickets': wickets,
            'overs': overs,
            'final_score': final_score,
            'checkpoint': checkpoint
        })

df_features = pd.DataFrame(features)
df_features.to_csv('data/ipl_processed_data.csv', index=False)
print("✅ Processed data saved to 'ipl_processed_data.csv'")
