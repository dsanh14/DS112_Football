#!/usr/bin/env python3
import json
import os
import re

print("Fixing the VAR LLM Analysis notebook to match your actual data columns...")

# Read the existing notebook
try:
    with open('4_VAR_LLM_Analysis.ipynb', 'r') as f:
        notebook = json.load(f)
except FileNotFoundError:
    print("Error: The notebook file '4_VAR_LLM_Analysis.ipynb' was not found.")
    print("Make sure you're running this script from the correct directory.")
    exit(1)
except Exception as e:
    print(f"Error reading notebook: {str(e)}")
    exit(1)

# Function to update code cells to match actual data
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
        
        # Incident description column
        ("df\\['incident_description'\\]", "df['Description']"),
        ("'incident_description'", "'Description'"),
        
        # Decision type column
        ("df\\['decision_type'\\]", "df['IncidentType']"),
        ("'decision_type'", "'IncidentType'"),
        
        # Decision column
        ("df\\['decision'\\]", "df['Decision']"),
        ("'decision'", "'Decision'"),
        
        # Team stats columns
        ("df\\['team_rank'\\]", "df['Rank']"),
        ("'team_rank'", "'Rank'"),
        
        # Opponent column
        ("df\\['opponent_name'\\]", "df['Opposition']"),
        ("'opponent_name'", "'Opposition'"),
        
        # Incident ID column
        ("df\\['incident_id'\\]", "df['IncidentID']"),
        ("'incident_id'", "'IncidentID'"),
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

# Update all code cells in the notebook
updated_cells = []
for cell in notebook['cells']:
    updated_cell = update_code_cell(cell)
    updated_cells.append(updated_cell)

# Update the notebook with new cells
notebook['cells'] = updated_cells

# Write the updated notebook back to file
try:
    with open('4_VAR_LLM_Analysis.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    print("✅ Successfully updated 4_VAR_LLM_Analysis.ipynb to work with your data!")
    print("You can now run the notebook with your actual data.")
except Exception as e:
    print(f"Error writing notebook: {str(e)}") 