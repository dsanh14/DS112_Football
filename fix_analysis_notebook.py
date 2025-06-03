#!/usr/bin/env python3
import json
import os
import re

print("Fixing the VAR Data Analysis notebook to match your actual data columns...")

# Read the existing notebook
try:
    with open('3_VAR_Data_Analysis.ipynb', 'r') as f:
        notebook = json.load(f)
except FileNotFoundError:
    print("Error: The notebook file '3_VAR_Data_Analysis.ipynb' was not found.")
    print("Make sure you're running this script from the correct directory.")
    exit(1)
except Exception as e:
    print(f"Error reading notebook: {str(e)}")
    exit(1)

# Function to update code cells to match actual data columns
def update_code_cell(cell):
    if cell['cell_type'] != 'code':
        return cell
    
    # The source in Jupyter notebooks is a list of strings
    source_lines = cell['source']
    
    # Replace column names to match actual data
    replacements = [
        # Team column
        ("df\\['team_name'\\]", "df['Team']"),
        ("'team_name'", "'Team'"),
        
        # Decision type column
        ("df\\['decision_type'\\]", "df['IncidentType']"),
        ("'decision_type'", "'IncidentType'"),
        
        # Decision favorable - we'll need to create this
        ("df\\['decision_favorable'\\]", "df['decision_favorable']"),
        
        # Team stats columns
        ("df\\['team_rank'\\]", "df['Rank']"),
        ("'team_rank'", "'Rank'"),
        ("df\\['market_value'\\]", "df['Goals_For']"),  # Using Goals_For as a proxy
        ("'market_value'", "'Goals_For'"),
        ("df\\['avg_attendance'\\]", "df['Fouls_Per_Game']"),  # Using Fouls_Per_Game as a proxy
        ("'avg_attendance'", "'Fouls_Per_Game'"),
        ("df\\['historical_success'\\]", "df['Wins']"),  # Using Wins as a proxy
        ("'historical_success'", "'Wins'"),
        
        # Time column
        ("df\\['match_minute'\\]", "df['TimeInMatch']"),
        ("'match_minute'", "'TimeInMatch'"),
        
        # Team tier - we'll need to create this
        ("df\\['team_tier'\\]", "df['team_tier']"),
    ]
    
    # Apply all replacements to each line
    updated_source = []
    for line in source_lines:
        updated_line = line
        for old, new in replacements:
            updated_line = re.sub(old, new, updated_line)
        updated_source.append(updated_line)
    
    # Update the cell source
    cell['source'] = updated_source
    return cell

# Add a cell to create necessary derived columns
def create_feature_engineering_cell():
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create necessary derived features to match the expected columns\n",
            "print(\"Creating derived features...\")\n",
            "\n",
            "# 1. Create decision_favorable column based on Decision and IncidentType\n",
            "def determine_favorability(incident_type, decision):\n",
            "    # Define favorable incidents\n",
            "    favorable_incidents = ['Penalty', 'Goal Review']\n",
            "    unfavorable_incidents = ['Red Card', 'Offside', 'Handball']\n",
            "    \n",
            "    # Define favorable decisions\n",
            "    favorable_decisions = ['Upheld']\n",
            "    unfavorable_decisions = ['Overturned']\n",
            "    \n",
            "    # Logic for favorability\n",
            "    if incident_type in favorable_incidents and decision in favorable_decisions:\n",
            "        return 1  # Favorable\n",
            "    elif incident_type in unfavorable_incidents and decision in unfavorable_decisions:\n",
            "        return 1  # Favorable (overturning a red card or offside is favorable)\n",
            "    elif incident_type in favorable_incidents and decision in unfavorable_decisions:\n",
            "        return 0  # Unfavorable\n",
            "    elif incident_type in unfavorable_incidents and decision in favorable_decisions:\n",
            "        return 0  # Unfavorable (upholding a red card or offside is unfavorable)\n",
            "    else:\n",
            "        return 0.5  # Neutral\n",
            "\n",
            "# Apply the function to create decision_favorable column\n",
            "df['decision_favorable'] = df.apply(lambda row: determine_favorability(row['IncidentType'], row['Decision']), axis=1)\n",
            "\n",
            "# 2. Create team_tier column based on Rank\n",
            "# Define quartiles for team ranking\n",
            "rank_bins = [0, 5, 10, 15, 20]  # Adjust based on your data\n",
            "rank_labels = ['Top Tier', 'Upper Mid', 'Lower Mid', 'Bottom Tier']\n",
            "\n",
            "# Create team_tier column\n",
            "df['team_tier'] = pd.cut(df['Rank'], bins=rank_bins, labels=rank_labels, include_lowest=True)\n",
            "\n",
            "# 3. Clean up TimeInMatch column to extract numeric minutes\n",
            "# Some TimeInMatch values might have format like '45+2'' or '90+3''\n",
            "def extract_minutes(time_str):\n",
            "    if pd.isna(time_str):\n",
            "        return 45  # Default to middle of game if missing\n",
            "    \n",
            "    # Remove any non-numeric characters and get the base minute\n",
            "    time_str = str(time_str).strip(\"'\")\n",
            "    if '+' in time_str:\n",
            "        base_minute = int(time_str.split('+')[0])\n",
            "        return base_minute\n",
            "    try:\n",
            "        return int(time_str.strip(\"'\"))\n",
            "    except:\n",
            "        return 45  # Default to middle of game if parsing fails\n",
            "\n",
            "# Apply the function if TimeInMatch is not already numeric\n",
            "if not pd.api.types.is_numeric_dtype(df['TimeInMatch']):\n",
            "    df['TimeInMatch'] = df['TimeInMatch'].apply(extract_minutes)\n",
            "\n",
            "print(\"✅ Derived features created successfully\")"
        ]
    }

# Update all code cells in the notebook
updated_cells = []

# First, add the base cells (title and imports)
for i, cell in enumerate(notebook['cells']):
    if i < 3:  # Keep the first 3 cells unchanged
        updated_cells.append(cell)
    else:
        break

# Add our feature engineering cell right after the imports
updated_cells.append(create_feature_engineering_cell())

# Add the rest of the cells with updated column names
for i, cell in enumerate(notebook['cells']):
    if i >= 3:  # Skip the first 3 cells we already added
        updated_cell = update_code_cell(cell)
        updated_cells.append(updated_cell)

# Update the notebook with new cells
notebook['cells'] = updated_cells

# Write the updated notebook back to file
try:
    with open('3_VAR_Data_Analysis.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    print("✅ Successfully updated 3_VAR_Data_Analysis.ipynb to work with your data!")
    print("You can now run the notebook with your actual data.")
except Exception as e:
    print(f"Error writing notebook: {str(e)}") 