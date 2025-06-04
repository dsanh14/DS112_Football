# VAR Fairness Audit: Detecting Bias in Football Officiating
**A Comprehensive Data Science Investigation | DS 112 Final Project**

*Diego Sanchez | Stanford University | Department of Data Science*

## Project Overview
This project presents a comprehensive analysis of potential bias in Video Assistant Referee (VAR) decisions in professional football. Using statistical methods, machine learning, and advanced visualization techniques, we analyzed 202 VAR incidents from the 2023-2024 season to investigate whether team characteristics influence decision favorability.

**Key Finding**: Top-tier teams received 15% more favorable decisions than bottom-tier teams, with penalty decisions showing a striking 32.8% disparity.

## Research Questions
1. Do higher-ranked teams receive more favorable VAR decisions?
2. Does the incident type affect the likelihood of favorable decisions?
3. Is there evidence of systemic bias in VAR decision-making over time?
4. What team characteristics best predict favorable VAR decisions?

## Dataset Description
- **202 VAR incidents** from Premier League, Championship, and League One
- **Time period**: 2023-2024 season
- **Incident types**: Red Card, Penalty, Goal Review, Offside, Handball
- **Decision outcomes**: Upheld, Overturned, No Clear Error
- **Team metrics**: Rank, Wins, Goals, Fouls/Cards per game
- **Source**: `var-fairness-audit/data/var_combined.csv`

## Analysis Notebooks

### 1. Data Extraction (`1_VAR_Data_Extraction.ipynb`)
- Data loading and cleaning procedures
- Feature engineering and data validation
- Creation of `decision_favorable` metric
- Robust error handling and data quality checks

### 2. Data Exploration (`2_VAR_Data_Exploration.ipynb`)
- Comprehensive visualization of VAR decision patterns
- Team tier analysis and ranking correlations
- Time series analysis of decision trends
- Distribution analysis by incident type and league

### 3. Data Analysis (`3_VAR_Data_Analysis.ipynb`)
- Statistical significance testing with bootstrap analysis
- Machine learning models for prediction
- Feature importance analysis using Random Forest
- Causal inference with propensity score matching

### 4. LLM Analysis (`4_VAR_LLM_Analysis.ipynb`)
- Google Gemini API integration for incident description analysis
- AI-powered bias detection in textual descriptions
- Sentiment analysis of VAR decision contexts

### 5. Complete Workflow (`VAR_Fairness_Audit_Complete.ipynb`)
- End-to-end analysis pipeline
- Consolidated findings and visualizations

## Key Findings

### Statistical Results
- **Overall disparity**: 15% more favorable decisions for top-tier teams
- **Penalty decisions**: 32.8% difference between top and bottom tiers
- **Strongest predictor**: Goals scored (offensive capability)
- **Bootstrap analysis**: Observed differences approach statistical significance

### Visualizations Created
1. **Team Rank vs. Decision Favorability**: Scatter plot with correlation analysis
2. **Decision Types by Team Tier**: Grouped bar chart showing incident-specific disparities
3. **Time Series Analysis**: Seasonal consistency of bias patterns
4. **Feature Importance**: Machine learning-derived predictive factors
5. **Bootstrap Analysis**: Statistical significance testing visualization

## Methodology

### Statistical Techniques
- **Bootstrap confidence intervals** (10,000 resamples)
- **Pearson correlation analysis**
- **T-tests and chi-square tests**
- **Time series analysis**

### Machine Learning
- **Random Forest classification** for feature importance
- **Logistic regression** for decision prediction
- **Cross-validation** for model robustness

### Advanced Analysis
- **Propensity score matching** for causal inference
- **Permutation testing** for validation
- **Network analysis** of referee-team interactions

## Research Poster
A comprehensive research poster was created following academic standards, featuring:
- Abstract and research questions
- Data exploration visualizations
- Statistical analysis results
- Conclusions and future directions
- Professional design optimized for conference presentation

## Presentation
A 5-10 minute passionate presentation script was developed, incorporating:
- Personal motivation and narrative storytelling
- Clear explanation of methodology and findings
- Discussion of implications for football fairness
- Call to action for continued research

## Technical Requirements
```python
# Core packages
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
scikit-learn >= 1.0.0

# Optional for LLM analysis
google-generativeai >= 0.3.0
```

## Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run notebooks in order**: Start with `1_VAR_Data_Extraction.ipynb`
4. **For LLM analysis**: Configure Google Gemini API key
5. **Generate visualizations**: All code included in notebooks

## Project Structure
```
DS112_Football/
├── 1_VAR_Data_Extraction.ipynb
├── 2_VAR_Data_Exploration.ipynb
├── 3_VAR_Data_Analysis.ipynb
├── 4_VAR_LLM_Analysis.ipynb
├── VAR_Fairness_Audit_Complete.ipynb
├── var-fairness-audit/
│   └── data/
│       └── var_combined.csv
└── README.md
```

## Future Directions
- Expand to multiple seasons for longitudinal analysis
- Incorporate referee-specific bias patterns
- Develop real-time VAR decision monitoring system
- Create standardized fairness audit protocol
- Apply to other sports with video review technology

## Acknowledgments
Special thanks to the DS 112 teaching staff for guidance and the open-source community for statistical analysis tools and visualization libraries.

## Contact
Diego Sanchez | dsanh14@stanford.edu

---
*This project demonstrates the intersection of sports analytics, fairness research, and data science methodology. The findings contribute to ongoing discussions about technology's role in maintaining competitive equity.*
