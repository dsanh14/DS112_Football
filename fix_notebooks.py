#!/usr/bin/env python3
import json
import os

# Function to create specialized notebooks
def create_notebook(title, description, specialized_cells):
    # Basic cells that appear in all notebooks
    base_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n\n**DS 112 Final Project**\n\n{description}"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages\n!pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Set plotting style\nplt.style.use('default')\nplt.rcParams['figure.figsize'] = (10, 6)"
            ]
        }
    ]
    
    # Combine base cells with specialized cells
    all_cells = base_cells + specialized_cells
    
    return {
        "cells": all_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            },
            "colab": {
                "provenance": []
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# Specialized cells for each notebook
extraction_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Loading\n\nFirst, we'll load the VAR incident and team stats datasets."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load VAR incident data\nvar_incidents = pd.read_csv('VAR_Incidents_Stats.csv')\n\n# Load team stats data\nteam_stats = pd.read_csv('VAR_Team_Stats.csv')\n\n# Display the first few rows of each dataset\nprint(\"VAR Incidents Dataset:\")\nvar_incidents.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display team stats dataset\nprint(\"Team Stats Dataset:\")\nteam_stats.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Cleaning\n\nNext, we'll clean the datasets by handling missing values and standardizing formats."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check for missing values\nprint(\"Missing values in VAR Incidents:\")\nprint(var_incidents.isnull().sum())\n\nprint(\"\\nMissing values in Team Stats:\")\nprint(team_stats.isnull().sum())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Merging\n\nNow we'll merge the two datasets to create a comprehensive dataset for analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Merge datasets on team name\nvar_combined = pd.merge(var_incidents, team_stats, on='team_name', how='left')\n\n# Display the merged dataset\nprint(\"Combined Dataset:\")\nvar_combined.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save the merged dataset\nvar_combined.to_csv('var_combined.csv', index=False)\nprint(\"Combined dataset saved to 'var_combined.csv'\")"
        ]
    }
]

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
            "# Load the combined dataset\ndf = pd.read_csv('var_combined.csv')\n\n# Display basic information\nprint(\"Dataset shape:\", df.shape)\ndf.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Basic Statistics\n\nLet's examine some basic statistics about VAR decisions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Count of VAR decisions by type\ndecision_counts = df['decision_type'].value_counts()\nprint(\"VAR Decision Counts:\")\nprint(decision_counts)\n\n# Create a pie chart of decisions\nplt.figure(figsize=(10, 6))\ndecision_counts.plot.pie(autopct='%1.1f%%', startangle=90)\nplt.title('Distribution of VAR Decisions')\nplt.ylabel('')\nplt.show()"
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
            "# Count of VAR decisions by team\nteam_decisions = df.groupby('team_name')['decision_type'].count().sort_values(ascending=False)\n\n# Plot teams with most VAR decisions\nplt.figure(figsize=(12, 6))\nteam_decisions.head(10).plot(kind='bar')\nplt.title('Teams with Most VAR Decisions')\nplt.xlabel('Team')\nplt.ylabel('Number of VAR Decisions')\nplt.xticks(rotation=45, ha='right')\nplt.tight_layout()\nplt.show()"
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
            "# Assuming there's a 'match_minute' column\n# Create bins for match time\ntime_bins = [0, 15, 30, 45, 60, 75, 90, 120]\ntime_labels = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '90+']\n\n# Add a match time bin column\ndf['time_period'] = pd.cut(df['match_minute'], bins=time_bins, labels=time_labels, right=False)\n\n# Count decisions by time period\ntime_decisions = df['time_period'].value_counts().sort_index()\n\n# Plot decisions by time period\nplt.figure(figsize=(12, 6))\ntime_decisions.plot(kind='bar')\nplt.title('VAR Decisions by Match Time')\nplt.xlabel('Match Time (minutes)')\nplt.ylabel('Number of VAR Decisions')\nplt.xticks(rotation=0)\nplt.tight_layout()\nplt.show()"
        ]
    }
]

analysis_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Load and Prepare Data\n\nFirst, let's load the combined dataset and prepare it for statistical analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load the combined dataset\ndf = pd.read_csv('var_combined.csv')\n\n# Display basic information\nprint(\"Dataset shape:\", df.shape)\ndf.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Statistical Tests\n\nLet's perform some statistical tests to check for potential bias in VAR decisions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import statistical libraries\nfrom scipy import stats\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report, confusion_matrix"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Chi-Square Test\n\nLet's test if there's a relationship between team ranking and favorable VAR decisions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create a contingency table (team rank vs. favorable decisions)\n# Assuming 'team_rank' and 'decision_favorable' columns exist\n\n# Bin teams into tiers based on ranking\ndf['team_tier'] = pd.qcut(df['team_rank'], q=4, labels=['Top Tier', 'Upper Mid', 'Lower Mid', 'Bottom Tier'])\n\n# Create contingency table\ncontingency = pd.crosstab(df['team_tier'], df['decision_favorable'])\nprint(\"Contingency Table:\")\nprint(contingency)\n\n# Chi-square test\nchi2, p, dof, expected = stats.chi2_contingency(contingency)\nprint(f\"\\nChi-square statistic: {chi2:.4f}\")\nprint(f\"p-value: {p:.4f}\")\nprint(f\"Degrees of freedom: {dof}\")\nprint(\"\\nExpected frequencies:\")\nprint(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Logistic Regression Model\n\nLet's build a model to predict favorable VAR decisions based on team characteristics."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Prepare features and target variable\nX = df[['team_rank', 'market_value', 'avg_attendance', 'historical_success']]\ny = df['decision_favorable']\n\n# Split data into train and test sets\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n\n# Train logistic regression model\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\n\n# Get feature coefficients\ncoefs = pd.DataFrame({\n    'Feature': X.columns,\n    'Coefficient': model.coef_[0]\n})\ncoefs = coefs.sort_values('Coefficient', ascending=False)\n\n# Plot coefficients\nplt.figure(figsize=(10, 6))\nsns.barplot(x='Coefficient', y='Feature', data=coefs)\nplt.title('Feature Importance for Predicting Favorable VAR Decisions')\nplt.axvline(x=0, color='black', linestyle='--')\nplt.tight_layout()\nplt.show()"
        ]
    }
]

llm_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Setup Google Gemini API\n\nFirst, let's set up the Google Gemini API for analyzing VAR incident descriptions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install the Google Generative AI library\n!pip install google-generativeai"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import necessary libraries\nimport google.generativeai as genai\nimport time\n\n# Configure Gemini API (you'll need to provide your API key)\ntry:\n    from google.colab import userdata\n    api_key = userdata.get('GEMINI_API_KEY')\n    genai.configure(api_key=api_key)\n    print(\"API configured successfully!\")\nexcept:\n    print(\"To use Google Gemini API, add your API key to Colab secrets.\")\n    print(\"1. Get an API key from https://ai.google.dev/\")\n    print(\"2. Go to Runtime > Manage Secrets and add GEMINI_API_KEY\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Load Dataset\n\nLet's load the VAR dataset and prepare the incident descriptions for analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load the combined dataset\ndf = pd.read_csv('var_combined.csv')\n\n# Display the first few incident descriptions\nprint(\"Sample VAR Incident Descriptions:\")\ndf[['incident_description', 'decision_type']].head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Analyze Incident Descriptions\n\nLet's use Google Gemini to analyze VAR incident descriptions for potential bias."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Function to analyze an incident description\ndef analyze_incident(incident_text, decision_type):\n    # Configure the model\n    model = genai.GenerativeModel('gemini-pro')\n    \n    # Create the prompt\n    prompt = f\"\"\"Analyze this soccer/football VAR (Video Assistant Referee) incident description objectively:\n    \n    Incident: {incident_text}\n    Official Decision: {decision_type}\n    \n    Based solely on this description, please evaluate:\n    1. Was this decision justified based on the rules of soccer?\n    2. Rate the controversy level of this decision (1-10 scale)\n    3. Is there any potential for bias in this decision?\n    \n    Provide your analysis in a structured format with clear reasoning.\"\"\"\n    \n    # Generate response\n    response = model.generate_content(prompt)\n    return response.text\n\n# Analyze a sample incident\nsample_idx = 0  # Change this to analyze different incidents\nsample_incident = df.iloc[sample_idx]\nprint(f\"Analyzing incident: {sample_incident['incident_description']}\")\nprint(f\"Decision: {sample_incident['decision_type']}\\n\")\n\n# Call the analysis function\ntry:\n    analysis = analyze_incident(sample_incident['incident_description'], sample_incident['decision_type'])\n    print(\"Gemini Analysis:\")\n    print(analysis)\nexcept Exception as e:\n    print(f\"Error: {e}\")\n    print(\"Make sure you've added your API key to Colab secrets.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Batch Analysis\n\nLet's analyze multiple incidents and compile the results."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Function to analyze multiple incidents\ndef batch_analyze(incidents_df, n_samples=5):\n    # Sample incidents to analyze\n    if len(incidents_df) > n_samples:\n        samples = incidents_df.sample(n_samples, random_state=42)\n    else:\n        samples = incidents_df\n    \n    # Collect results\n    results = []\n    \n    # Analyze each sample\n    for i, incident in samples.iterrows():\n        try:\n            print(f\"Analyzing incident {i+1}/{len(samples)}...\")\n            analysis = analyze_incident(incident['incident_description'], incident['decision_type'])\n            \n            # Add to results\n            results.append({\n                'incident_id': i,\n                'description': incident['incident_description'],\n                'decision': incident['decision_type'],\n                'analysis': analysis\n            })\n            \n            # Sleep to avoid rate limiting\n            time.sleep(1)\n            \n        except Exception as e:\n            print(f\"Error analyzing incident {i}: {e}\")\n    \n    # Convert to DataFrame\n    return pd.DataFrame(results)\n\n# Uncomment to run batch analysis\n# Set n_samples to a small number for testing\n# analysis_results = batch_analyze(df, n_samples=3)\n# analysis_results"
        ]
    }
]

complete_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Complete VAR Fairness Audit\n\nThis notebook combines all elements of the VAR fairness audit into a single comprehensive analysis."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Loading and Preparation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load VAR incident data\ntry:\n    var_incidents = pd.read_csv('VAR_Incidents_Stats.csv')\n    team_stats = pd.read_csv('VAR_Team_Stats.csv')\n    print(\"Loaded raw datasets successfully\")\nexcept FileNotFoundError:\n    print(\"Raw data files not found. Using the combined dataset if available.\")\n\n# Try to load the combined dataset\ntry:\n    df = pd.read_csv('var_combined.csv')\n    print(\"Loaded combined dataset successfully\")\nexcept FileNotFoundError:\n    print(\"Creating combined dataset...\")\n    # Merge datasets\n    df = pd.merge(var_incidents, team_stats, on='team_name', how='left')\n    df.to_csv('var_combined.csv', index=False)\n    print(\"Created and saved combined dataset\")\n\n# Display dataset info\nprint(\"\\nDataset Information:\")\ndf.info()\n\nprint(\"\\nSample Data:\")\ndf.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Exploratory Data Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Overview of VAR decisions\nplt.figure(figsize=(12, 5))\nplt.subplot(1, 2, 1)\ndf['decision_type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)\nplt.title('Distribution of VAR Decisions')\nplt.ylabel('')\n\n# Decision outcomes by team tier\nplt.subplot(1, 2, 2)\n# Create team tiers based on ranking\ndf['team_tier'] = pd.qcut(df['team_rank'], q=4, labels=['Top Tier', 'Upper Mid', 'Lower Mid', 'Bottom Tier'])\n# Plot favorable decisions by team tier\nsns.countplot(x='team_tier', hue='decision_favorable', data=df)\nplt.title('Favorable Decisions by Team Tier')\nplt.xlabel('Team Tier')\nplt.ylabel('Count')\nplt.xticks(rotation=45)\nplt.tight_layout()\nplt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Statistical Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import statistical libraries\nfrom scipy import stats\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report\n\n# Chi-square test of independence\ncontingency = pd.crosstab(df['team_tier'], df['decision_favorable'])\nchi2, p, dof, expected = stats.chi2_contingency(contingency)\nprint(f\"Chi-square Test Results:\")\nprint(f\"Chi-square statistic: {chi2:.4f}\")\nprint(f\"p-value: {p:.4f}\")\nprint(f\"Degrees of freedom: {dof}\")\n\n# Logistic regression\nX = df[['team_rank', 'market_value', 'avg_attendance']]\ny = df['decision_favorable']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\nmodel = LogisticRegression()\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\n\n# Model evaluation\nprint(\"\\nLogistic Regression Results:\")\nprint(classification_report(y_test, y_pred))\n\n# Feature importance\ncoefs = pd.DataFrame({\n    'Feature': X.columns,\n    'Coefficient': model.coef_[0]\n})\ncoefs = coefs.sort_values('Coefficient', ascending=False)\nprint(\"\\nFeature Importance:\")\nprint(coefs)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. LLM Analysis\n\nFor a complete analysis, we also need to incorporate language model analysis of incident descriptions.\n\nNote: To run this section, you need to add your Gemini API key to Colab secrets."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install and import Google Generative AI library\ntry:\n    !pip install google-generativeai\n    import google.generativeai as genai\n    print(\"Google Generative AI library installed\")\nexcept:\n    print(\"Could not install Google Generative AI library\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configure the API (only if you want to run LLM analysis)\ntry:\n    from google.colab import userdata\n    api_key = userdata.get('GEMINI_API_KEY')\n    genai.configure(api_key=api_key)\n    print(\"API configured successfully\")\n    \n    # Sample function for incident analysis\n    def analyze_incident(incident_text, decision_type):\n        model = genai.GenerativeModel('gemini-pro')\n        prompt = f\"\"\"Analyze this soccer VAR incident objectively:\\n\\nIncident: {incident_text}\\nDecision: {decision_type}\\n\\nEvaluate: Was this decision justified? Rate controversy (1-10).\"\"\"\n        response = model.generate_content(prompt)\n        return response.text\n    \n    # Analyze a sample incident\n    sample = df.iloc[0]\n    print(f\"\\nSample Incident: {sample['incident_description']}\")\n    print(f\"Decision: {sample['decision_type']}\")\n    print(\"\\nAnalysis:\")\n    print(analyze_incident(sample['incident_description'], sample['decision_type']))\n    \nexcept Exception as e:\n    print(f\"To run LLM analysis, add your API key to Colab secrets: {e}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Conclusions\n\nSummarize the key findings from your VAR fairness audit analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize the key findings\nplt.figure(figsize=(10, 6))\n\n# Plot feature importance for predicting favorable decisions\nsns.barplot(x='Coefficient', y='Feature', data=coefs)\nplt.title('Factors Influencing Favorable VAR Decisions')\nplt.axvline(x=0, color='black', linestyle='--')\nplt.tight_layout()\nplt.show()\n\n# Print conclusion\nprint(\"VAR Fairness Audit Conclusions:\")\nprint(\"1. Statistical significance: The chi-square test p-value indicates whether team tier and decision favorability are independent.\")\nprint(f\"   p-value: {p:.4f} - {'Evidence of bias' if p < 0.05 else 'No strong evidence of bias'}\")\nprint(\"\\n2. Predictive modeling: We examined if team characteristics can predict favorable decisions.\")\nprint(f\"   The most influential factors are: {', '.join(coefs['Feature'].head(2).tolist())}\")\nprint(\"\\n3. Recommendations: Based on the analysis, VAR implementation could be improved by...\")"
        ]
    }
]

# Create notebooks
notebooks = [
    {
        "filename": "1_VAR_Data_Extraction.ipynb",
        "title": "VAR Fairness Audit: Data Extraction",
        "description": "This notebook focuses on loading and preparing the VAR datasets for analysis.",
        "specialized_cells": extraction_cells
    },
    {
        "filename": "2_VAR_Data_Exploration.ipynb",
        "title": "VAR Fairness Audit: Data Exploration",
        "description": "This notebook provides visualizations and exploratory analysis of VAR decisions.",
        "specialized_cells": exploration_cells
    },
    {
        "filename": "3_VAR_Data_Analysis.ipynb",
        "title": "VAR Fairness Audit: Statistical Analysis",
        "description": "This notebook performs statistical tests and ML modeling to detect potential bias.",
        "specialized_cells": analysis_cells
    },
    {
        "filename": "4_VAR_LLM_Analysis.ipynb",
        "title": "VAR Fairness Audit: LLM Analysis",
        "description": "This notebook uses Google Gemini to analyze VAR incident descriptions.",
        "specialized_cells": llm_cells
    },
    {
        "filename": "VAR_Fairness_Audit_Complete.ipynb",
        "title": "VAR Fairness Audit: Complete Analysis",
        "description": "This notebook provides a complete solution for analyzing VAR decisions in Google Colab.",
        "specialized_cells": complete_cells
    }
]

# Create each notebook
for nb in notebooks:
    notebook = create_notebook(nb["title"], nb["description"], nb["specialized_cells"])
    with open(nb["filename"], 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created specialized notebook: {nb['filename']}")

print("\n✅ All notebooks created successfully with specialized content!")
print("These notebooks can now be opened in Google Colab and will work as specialized templates for each analysis phase.") 