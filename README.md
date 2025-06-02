# VAR Fairness Audit Project - DS 112
**A Data & Language Model Investigation of Soccer Refereeing Bias**

## 📊 Project Overview
This project analyzes potential bias in Video Assistant Referee (VAR) decisions in professional soccer using both statistical methods and AI-powered techniques.

## 🗂️ Data Files
- **VAR_Incidents_Stats.csv**: Contains information about individual VAR incidents
- **VAR_Team_Stats.csv**: Team statistics and rankings 
- **var_combined.csv**: Merged dataset with all information (ready for analysis)

## 🚀 Running This Analysis in Google Colab

### Option 1: Starting from Scratch
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Upload the data files using the file browser (left sidebar)
4. Install required packages:
   ```python
   !pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy google-generativeai
   ```
5. Load the data:
   ```python
   import pandas as pd
   df = pd.read_csv('var_combined.csv')  # Or combine individual files
   ```

### Option 2: Creating Specialized Notebooks
For a complete analysis workflow, we recommend creating separate notebooks for:

1. **Data Extraction**: Loading, cleaning, and merging data
   - Key libraries: pandas, numpy
   - Input: VAR_Incidents_Stats.csv, VAR_Team_Stats.csv
   - Output: var_combined.csv

2. **Data Exploration**: Visualizing patterns and trends
   - Key libraries: matplotlib, seaborn, plotly
   - Create visualizations for incidents, decisions, team patterns

3. **Statistical Analysis**: Testing for significant bias
   - Key libraries: scipy, sklearn
   - Perform t-tests, chi-square tests, and machine learning models

4. **LLM Analysis**: Using Google Gemini to analyze descriptions
   - Key library: google-generativeai
   - API key needed from Google AI Studio

## 🔑 For LLM Analysis (Optional)
To use Google Gemini for analyzing incident descriptions:
1. Get an API key from [Google AI Studio](https://ai.google.dev/)
2. Add it to Colab secrets: Runtime > Manage Secrets
3. Configure the API:
   ```python
   import google.generativeai as genai
   from google.colab import userdata
   api_key = userdata.get('GEMINI_API_KEY')
   genai.configure(api_key=api_key)
   ```

## 📈 Analysis Components
Your DS 112 project can include any combination of:

- **Statistical Tests**: T-tests, Chi-square tests, Effect sizes
- **Machine Learning**: Logistic regression, Random forests
- **Visualizations**: Team comparisons, Decision patterns, Bias indicators
- **LLM Component**: AI-powered analysis of incident descriptions

## 💡 Project Structure Suggestion
```
VAR_Fairness_Audit/
├── data/
│   ├── VAR_Incidents_Stats.csv
│   ├── VAR_Team_Stats.csv
│   └── var_combined.csv
├── notebooks/
│   ├── 1_Data_Extraction.ipynb
│   ├── 2_Data_Exploration.ipynb
│   ├── 3_Statistical_Analysis.ipynb
│   └── 4_LLM_Analysis.ipynb
└── README.md
```

## ✅ DS 112 Requirements Coverage
This project structure demonstrates:
- **Technical Challenge**: Statistical analysis, ML modeling, LLM integration
- **Real-World Application**: Sports fairness analysis
- **Novelty**: First VAR bias analysis using combined methods
- **Comprehensive Analysis**: Multiple techniques with clear storytelling
