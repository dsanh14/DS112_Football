#!/usr/bin/env python3
import json
import os

# Read the existing notebook
with open('2_VAR_Data_Exploration.ipynb', 'r') as f:
    notebook = json.load(f)

# Enhanced cells for Data Exploration notebook
exploration_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Load Combined Dataset\n\nFirst, let's load the combined VAR dataset created in the previous notebook."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Output file from data extraction notebook\nCOMBINED_FILE = 'var_combined.csv'\n\n# Load the combined dataset\ntry:\n    df = pd.read_csv(COMBINED_FILE)\n    print(f\"✅ Successfully loaded dataset from {COMBINED_FILE}\")\n    print(f\"Dataset shape: {df.shape} (rows, columns)\")\n    print(\"\\nFirst 5 rows:\")\n    display(df.head())\n    \n    # Display column information\n    print(\"\\nColumn information:\")\n    df.info()\n    \n    # Basic statistics\n    print(\"\\nBasic statistics for numerical columns:\")\n    display(df.describe())\n    \nexcept FileNotFoundError:\n    print(f\"❌ Error: Could not find {COMBINED_FILE}\")\n    print(\"Please run the Data Extraction notebook first.\")\nexcept Exception as e:\n    print(f\"❌ Error loading data: {str(e)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Distribution Analysis\n\nLet's examine the distribution of VAR decisions and related variables."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyze VAR decision types\nif 'decision_type' in df.columns:\n    decision_counts = df['decision_type'].value_counts()\n    print(\"VAR Decision Type Counts:\")\n    print(decision_counts)\n    \n    # Create a pie chart of decisions\n    plt.figure(figsize=(12, 6))\n    plt.subplot(1, 2, 1)\n    decision_counts.plot.pie(autopct='%1.1f%%', startangle=90)\n    plt.title('Distribution of VAR Decisions')\n    plt.ylabel('')\n    \n    # Create a bar chart of decisions\n    plt.subplot(1, 2, 2)\n    decision_counts.plot(kind='bar', color='skyblue')\n    plt.title('VAR Decision Counts')\n    plt.xlabel('Decision Type')\n    plt.ylabel('Count')\n    plt.xticks(rotation=45, ha='right')\n    plt.tight_layout()\n    plt.show()\n    \n    # Plot favorable vs unfavorable decisions\n    if 'decision_favorable' in df.columns:\n        favorable_counts = df['decision_favorable'].value_counts()\n        plt.figure(figsize=(8, 5))\n        favorable_counts.plot.pie(autopct='%1.1f%%', startangle=90, \n                                colors=['#ff9999','#66b3ff','#99ff99'])\n        plt.title('Distribution of Favorable vs. Unfavorable Decisions')\n        plt.ylabel('')\n        plt.show()\nelse:\n    print(\"⚠️ 'decision_type' column not found in the dataset\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Team Analysis\n\nLet's explore how VAR decisions are distributed among teams."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyze VAR decisions by team\nif 'team_name' in df.columns:\n    # Count decisions by team\n    team_decisions = df.groupby('team_name').size().sort_values(ascending=False)\n    \n    # Plot teams with most VAR decisions\n    plt.figure(figsize=(14, 6))\n    team_decisions.head(15).plot(kind='bar', color='skyblue')\n    plt.title('Teams with Most VAR Decisions')\n    plt.xlabel('Team')\n    plt.ylabel('Number of VAR Decisions')\n    plt.xticks(rotation=45, ha='right')\n    plt.tight_layout()\n    plt.show()\n    \n    # Analyze favorable decisions by team\n    if 'decision_favorable' in df.columns:\n        # Calculate percentage of favorable decisions by team\n        team_favor = df.groupby('team_name')['decision_favorable'].mean().sort_values(ascending=False)\n        \n        # Plot teams by favorable decision percentage\n        plt.figure(figsize=(14, 6))\n        team_favor.head(15).plot(kind='bar', color='lightgreen')\n        plt.title('Teams with Highest Percentage of Favorable VAR Decisions')\n        plt.xlabel('Team')\n        plt.ylabel('Favorable Decision Percentage')\n        plt.axhline(y=team_favor.mean(), color='red', linestyle='--', \n                   label=f'Average ({team_favor.mean():.2f})')\n        plt.xticks(rotation=45, ha='right')\n        plt.legend()\n        plt.tight_layout()\n        plt.show()\n        \n        # Create a team tier analysis\n        if 'team_tier' in df.columns:\n            # Calculate favorable decisions by team tier\n            tier_favor = df.groupby('team_tier')['decision_favorable'].mean().sort_values(ascending=False)\n            \n            # Plot favorable decisions by team tier\n            plt.figure(figsize=(10, 5))\n            tier_favor.plot(kind='bar', color='orange')\n            plt.title('Favorable VAR Decision Percentage by Team Tier')\n            plt.xlabel('Team Tier')\n            plt.ylabel('Favorable Decision Percentage')\n            plt.axhline(y=df['decision_favorable'].mean(), color='red', linestyle='--', \n                       label=f'Overall Average ({df[\"decision_favorable\"].mean():.2f})')\n            plt.xticks(rotation=0)\n            plt.legend()\n            plt.tight_layout()\n            plt.show()\nelse:\n    print(\"⚠️ 'team_name' column not found in the dataset\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Time Analysis\n\nDo VAR decisions happen more frequently at certain times during matches?"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyze VAR decisions by match time\nif 'match_minute' in df.columns:\n    # Create a histogram of decision times\n    plt.figure(figsize=(12, 6))\n    plt.hist(df['match_minute'], bins=18, color='skyblue', edgecolor='black')\n    plt.title('Distribution of VAR Decisions by Match Minute')\n    plt.xlabel('Match Minute')\n    plt.ylabel('Number of VAR Decisions')\n    plt.grid(axis='y', alpha=0.75)\n    plt.show()\n    \n    # Use time_period if available, otherwise create it\n    if 'time_period' not in df.columns:\n        # Create time periods\n        time_bins = [0, 15, 30, 45, 60, 75, 90, 120]\n        time_labels = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '90+']\n        df['time_period'] = pd.cut(df['match_minute'], \n                                  bins=time_bins, \n                                  labels=time_labels, \n                                  right=False)\n    \n    # Count decisions by time period\n    time_decisions = df['time_period'].value_counts().sort_index()\n    \n    # Plot decisions by time period\n    plt.figure(figsize=(12, 6))\n    time_decisions.plot(kind='bar', color='lightgreen')\n    plt.title('VAR Decisions by Match Time Period')\n    plt.xlabel('Match Time (minutes)')\n    plt.ylabel('Number of VAR Decisions')\n    plt.xticks(rotation=0)\n    plt.grid(axis='y', alpha=0.3)\n    plt.tight_layout()\n    plt.show()\n    \n    # Analyze decision types by time period\n    if 'decision_type' in df.columns:\n        # Create a crosstab of time period vs decision type\n        time_decision_cross = pd.crosstab(df['time_period'], df['decision_type'])\n        \n        # Plot stacked bar chart\n        plt.figure(figsize=(14, 7))\n        time_decision_cross.plot(kind='bar', stacked=True, figsize=(14, 7))\n        plt.title('Types of VAR Decisions by Match Time Period')\n        plt.xlabel('Match Time Period')\n        plt.ylabel('Number of Decisions')\n        plt.legend(title='Decision Type', bbox_to_anchor=(1.05, 1), loc='upper left')\n        plt.tight_layout()\n        plt.show()\n        \n        # Analyze favorability by time period\n        if 'decision_favorable' in df.columns:\n            time_favor = df.groupby('time_period')['decision_favorable'].mean()\n            \n            plt.figure(figsize=(12, 5))\n            time_favor.plot(kind='bar', color='orange')\n            plt.title('Favorable Decision Percentage by Match Time Period')\n            plt.xlabel('Match Time Period')\n            plt.ylabel('Favorable Decision Percentage')\n            plt.axhline(y=df['decision_favorable'].mean(), color='red', linestyle='--', \n                       label=f'Overall Average ({df[\"decision_favorable\"].mean():.2f})')\n            plt.xticks(rotation=0)\n            plt.legend()\n            plt.grid(axis='y', alpha=0.3)\n            plt.tight_layout()\n            plt.show()\nelse:\n    print(\"⚠️ 'match_minute' column not found in the dataset\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Correlation Analysis\n\nLet's examine relationships between team characteristics and VAR decisions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Identify numerical columns for correlation analysis\nnum_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()\n\n# Remove any ID columns or other irrelevant columns\nnum_cols = [col for col in num_cols if not ('id' in col.lower() or 'index' in col.lower())]\n\nif len(num_cols) > 1:\n    # Calculate correlations\n    corr = df[num_cols].corr()\n    \n    # Plot correlation heatmap\n    plt.figure(figsize=(12, 10))\n    mask = np.triu(np.ones_like(corr, dtype=bool))\n    cmap = sns.diverging_palette(230, 20, as_cmap=True)\n    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,\n                square=True, linewidths=.5, cbar_kws={\"shrink\": .5}, annot=True)\n    plt.title('Correlation Matrix of Numerical Variables')\n    plt.tight_layout()\n    plt.show()\n    \n    # Plot key relationships if decision_favorable exists\n    if 'decision_favorable' in num_cols:\n        # Find top correlated features with decision_favorable\n        decision_corr = corr['decision_favorable'].sort_values(ascending=False)\n        print(\"Correlations with 'decision_favorable':\")\n        print(decision_corr)\n        \n        # Plot the top 3 correlations (excluding self-correlation)\n        top_corr = decision_corr[1:4]\n        for col in top_corr.index:\n            plt.figure(figsize=(10, 6))\n            plt.scatter(df[col], df['decision_favorable'], alpha=0.5)\n            plt.title(f'Relationship: {col} vs. Favorable Decisions')\n            plt.xlabel(col)\n            plt.ylabel('Decision Favorability')\n            plt.grid(alpha=0.3)\n            plt.show()\nelse:\n    print(\"⚠️ Not enough numerical columns for correlation analysis\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Key Findings\n\nSummarize the main patterns and insights discovered in the exploratory analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create a summary of key findings\nprint(\"Key Findings from Exploratory Analysis:\")\nprint(\"========================================\")\n\n# 1. Decision distribution\nif 'decision_type' in df.columns:\n    most_common = df['decision_type'].value_counts().index[0]\n    print(f\"1. Most common VAR decision: {most_common} ({df['decision_type'].value_counts().iloc[0]} occurrences)\")\n\n# 2. Team with most decisions\nif 'team_name' in df.columns:\n    team_counts = df['team_name'].value_counts()\n    top_team = team_counts.index[0]\n    print(f\"2. Team with most VAR decisions: {top_team} ({team_counts.iloc[0]} decisions)\")\n\n# 3. Team with highest favorable percentage\nif 'team_name' in df.columns and 'decision_favorable' in df.columns:\n    # Filter to teams with at least 5 decisions\n    team_min_decisions = 5\n    team_decisions = df['team_name'].value_counts()\n    teams_with_min = team_decisions[team_decisions >= team_min_decisions].index.tolist()\n    \n    if teams_with_min:\n        team_favor = df[df['team_name'].isin(teams_with_min)].groupby('team_name')['decision_favorable'].mean()\n        most_favored = team_favor.sort_values(ascending=False).index[0]\n        favor_pct = team_favor.sort_values(ascending=False).iloc[0]\n        print(f\"3. Team with highest favorable decision %: {most_favored} ({favor_pct:.1%})\")\n\n# 4. Time period with most decisions\nif 'time_period' in df.columns:\n    time_counts = df['time_period'].value_counts()\n    busy_time = time_counts.index[0]\n    print(f\"4. Match time with most VAR decisions: {busy_time} ({time_counts.iloc[0]} decisions)\")\n\n# 5. Potential bias indicators\nif 'team_tier' in df.columns and 'decision_favorable' in df.columns:\n    tier_favor = df.groupby('team_tier')['decision_favorable'].mean()\n    print(\"5. Favorable decision % by team tier:\")\n    for tier, value in tier_favor.items():\n        print(f\"   - {tier}: {value:.1%}\")\n\nprint(\"\\nNext steps: Statistical testing to confirm these patterns\")"
        ]
    }
]

# Replace cells in the notebook with our enhanced cells
# First, keep the first 3 cells (title and imports)
base_cells = notebook['cells'][:3]
# Combine with our enhanced cells
notebook['cells'] = base_cells + exploration_cells

# Write the enhanced notebook back to file
with open('2_VAR_Data_Exploration.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Enhanced the Data Exploration notebook with more extensive code!") 