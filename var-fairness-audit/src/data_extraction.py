def extract_data():
    import pandas as pd
    import os

    incidents_path = "data/VAR_Incidents_Stats.csv"
    team_stats_path = "data/VAR_Team_Stats.csv"

    incidents_df = pd.read_csv(incidents_path)
    team_stats_df = pd.read_csv(team_stats_path)

    # Handle missing values
    incidents_df.fillna("Unknown", inplace=True)
    team_stats_df.fillna("Unknown", inplace=True)

    # Merge on common column
    merged_df = pd.merge(incidents_df, team_stats_df, on="Team", how="inner")

    # Create data/ folder if not exists
    os.makedirs("data", exist_ok=True)
    merged_df.to_csv("data/var_combined.csv", index=False)

    print("✅ Data extracted and saved to data/var_combined.csv") 