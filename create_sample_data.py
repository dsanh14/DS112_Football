#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Create sample VAR incidents data
def create_var_incidents():
    # Define teams
    teams = [
        'Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 
        'Manchester City', 'Tottenham', 'Leicester City', 'Everton',
        'West Ham', 'Aston Villa', 'Newcastle', 'Southampton',
        'Brighton', 'Wolves', 'Crystal Palace', 'Burnley',
        'Leeds United', 'Watford', 'Norwich City', 'Brentford'
    ]
    
    # Define decision types
    decision_types = [
        'goal_disallowed', 'goal_allowed', 'penalty_awarded', 
        'penalty_overturned', 'red_card_to_team', 'red_card_to_opponent'
    ]
    
    # Generate random data
    n_incidents = 300
    data = {
        'incident_id': range(1, n_incidents + 1),
        'team_name': [random.choice(teams) for _ in range(n_incidents)],
        'opponent_name': [random.choice(teams) for _ in range(n_incidents)],
        'decision_type': [random.choice(decision_types) for _ in range(n_incidents)],
        'match_minute': [random.randint(1, 95) for _ in range(n_incidents)],
        'season': [random.choice(['2019-20', '2020-21', '2021-22']) for _ in range(n_incidents)],
        'incident_description': [f"Incident description {i}" for i in range(n_incidents)]
    }
    
    # Ensure opponent is different from team
    for i in range(n_incidents):
        while data['opponent_name'][i] == data['team_name'][i]:
            data['opponent_name'][i] = random.choice(teams)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

# Create sample team stats data
def create_team_stats():
    # Define teams (same as in incidents)
    teams = [
        'Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 
        'Manchester City', 'Tottenham', 'Leicester City', 'Everton',
        'West Ham', 'Aston Villa', 'Newcastle', 'Southampton',
        'Brighton', 'Wolves', 'Crystal Palace', 'Burnley',
        'Leeds United', 'Watford', 'Norwich City', 'Brentford'
    ]
    
    # Generate random data
    n_teams = len(teams)
    ranks = list(range(1, n_teams + 1))
    random.shuffle(ranks)
    
    data = {
        'team_name': teams,
        'team_rank': ranks,
        'market_value': [round(random.uniform(100, 1000), 1) for _ in range(n_teams)],
        'avg_attendance': [random.randint(30000, 75000) for _ in range(n_teams)],
        'historical_success': [random.randint(1, 10) for _ in range(n_teams)],
        'league_position': [random.randint(1, 20) for _ in range(n_teams)],
        'goals_scored': [random.randint(20, 80) for _ in range(n_teams)],
        'goals_conceded': [random.randint(20, 80) for _ in range(n_teams)]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

# Generate the datasets
print("Generating sample VAR incidents dataset...")
var_incidents = create_var_incidents()

print("Generating sample team stats dataset...")
team_stats = create_team_stats()

# Save to CSV
var_incidents.to_csv('VAR_Incidents_Stats.csv', index=False)
team_stats.to_csv('VAR_Team_Stats.csv', index=False)

print(f"\n✅ Created VAR_Incidents_Stats.csv with {len(var_incidents)} incidents")
print(f"✅ Created VAR_Team_Stats.csv with {len(team_stats)} teams")
print("\nYou can now run the VAR analysis notebooks in sequence.") 