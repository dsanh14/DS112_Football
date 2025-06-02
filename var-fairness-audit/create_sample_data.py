"""
Sample Data Generator for VAR Fairness Audit

This script creates realistic sample datasets for testing the VAR analysis.
Run this if you don't have actual VAR data files.

Usage: python create_sample_data.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_data():
    """Generate realistic sample VAR incident and team data"""
    
    np.random.seed(42)  # For reproducible results
    
    # Sample team data
    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
        "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
        "Crystal Palace", "Wolves", "Fulham", "Brentford", "Nottingham Forest",
        "Everton", "Leeds United", "Leicester City", "Southampton", "Bournemouth"
    ]
    
    leagues = ["Premier League", "Championship", "League One"]
    
    # Create team stats
    team_stats_data = []
    for i, team in enumerate(teams):
        team_stats_data.append({
            'Team': team,
            'Rank': i + 1 + np.random.randint(-2, 3),  # Add some randomness
            'League': np.random.choice(leagues, p=[0.7, 0.2, 0.1]),
            'Wins': np.random.randint(10, 30),
            'Losses': np.random.randint(5, 15),
            'Goals_For': np.random.randint(30, 80),
            'Goals_Against': np.random.randint(20, 60),
            'Fouls_Per_Game': np.random.uniform(8, 18),
            'Cards_Per_Game': np.random.uniform(1.5, 4.0)
        })
    
    team_stats_df = pd.DataFrame(team_stats_data)
    
    # Create VAR incidents data
    incidents_data = []
    decision_types = ["Overturned", "Upheld", "No Clear Error"]
    incident_types = ["Penalty", "Offside", "Red Card", "Goal Review", "Handball"]
    
    # Generate incidents with some bias patterns for interesting analysis
    num_incidents = 200
    
    for i in range(num_incidents):
        team = np.random.choice(teams)
        team_rank = team_stats_df[team_stats_df['Team'] == team]['Rank'].iloc[0]
        
        # Introduce subtle bias - lower ranked teams slightly more likely to have overturns
        if team_rank <= 7:  # Top tier teams
            overturn_prob = 0.15
        elif team_rank <= 14:  # Mid tier
            overturn_prob = 0.25
        else:  # Bottom tier
            overturn_prob = 0.35
        
        decision = np.random.choice(decision_types, 
                                  p=[overturn_prob, 0.6, 0.4 - overturn_prob])
        
        # Generate incident date
        start_date = datetime(2023, 8, 1)
        incident_date = start_date + timedelta(days=np.random.randint(0, 300))
        
        incidents_data.append({
            'IncidentID': f"VAR_{i+1:03d}",
            'Team': team,
            'Opposition': np.random.choice([t for t in teams if t != team]),
            'Date': incident_date.strftime('%Y-%m-%d'),
            'Round': np.random.randint(1, 38),
            'IncidentType': np.random.choice(incident_types),
            'Decision': decision,
            'TimeInMatch': f"{np.random.randint(1, 90)}'",
            'RefereeName': f"Referee_{np.random.randint(1, 20)}",
            'Description': generate_incident_description(np.random.choice(incident_types)),
            'League': team_stats_df[team_stats_df['Team'] == team]['League'].iloc[0]
        })
    
    incidents_df = pd.DataFrame(incidents_data)
    
    return team_stats_df, incidents_df

def generate_incident_description(incident_type):
    """Generate realistic incident descriptions for LLM analysis"""
    
    descriptions = {
        "Penalty": [
            "Player fouled in penalty box, VAR reviewed for clear and obvious error",
            "Defender made contact with striker in box, penalty decision reviewed",
            "Handball in penalty area checked by VAR for intentional contact",
            "VAR intervention for missed penalty after goalkeeper contact"
        ],
        "Offside": [
            "Offside call questioned, VAR checked player position at time of pass",
            "Tight offside decision reviewed using VAR technology",
            "Goal disallowed for offside, VAR confirmed correct decision",
            "VAR overturned offside call after detailed review"
        ],
        "Red Card": [
            "Red card decision reviewed for violent conduct in midfield challenge",
            "Serious foul play checked by VAR for card upgrade",
            "VAR intervention for missed second yellow card in aggressive tackle",
            "Direct red card reviewed for endangering opponent safety"
        ],
        "Goal Review": [
            "Goal scored but VAR checked for possible handball in buildup play",
            "VAR reviewed goal for potential offside in attacking sequence",
            "Goal celebration halted for VAR check on foul in buildup",
            "Disputed goal reviewed for ball crossing goal line"
        ],
        "Handball": [
            "Handball incident in box reviewed for penalty decision",
            "VAR checked handball for unnatural arm position",
            "Ball struck arm of defender, VAR assessed for deliberate contact",
            "Handball in buildup to goal reviewed by VAR officials"
        ]
    }
    
    return np.random.choice(descriptions.get(incident_type, ["VAR incident reviewed"]))

def main():
    """Main function to create and save sample data"""
    
    print("🔄 Generating sample VAR data...")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate datasets
    team_stats_df, incidents_df = create_sample_data()
    
    # Save to CSV files
    team_stats_df.to_csv("data/VAR_Team_Stats.csv", index=False)
    incidents_df.to_csv("data/VAR_Incidents_Stats.csv", index=False)
    
    print("✅ Sample data created successfully!")
    print(f"📊 Generated {len(team_stats_df)} teams and {len(incidents_df)} VAR incidents")
    print("\nFiles created:")
    print("• data/VAR_Team_Stats.csv")
    print("• data/VAR_Incidents_Stats.csv")
    print("\n🚀 You can now run the main analysis: python main.py")
    
    # Display sample data preview
    print("\n📋 Sample Team Stats Preview:")
    print(team_stats_df.head().to_string())
    
    print("\n📋 Sample Incidents Preview:")
    print(incidents_df.head().to_string())

if __name__ == "__main__":
    main() 