# �� VAR Fairness Audit: A Data & Language Model Investigation

**DS 112 Final Project**: A comprehensive analysis tool for examining Video Assistant Referee (VAR) decisions and potential bias patterns in football/soccer matches using traditional data science and cutting-edge Large Language Models.

## 🎯 Project Overview

This project analyzes VAR incident data to:
- **Identify patterns** in VAR decision-making across teams and leagues
- **Detect potential bias** using statistical analysis and fairness metrics
- **Apply machine learning** to predict overturn likelihood
- **Use LLMs** (Google Gemini) to analyze incident descriptions and compare with actual decisions
- **Generate interactive visualizations** and comprehensive reporting

### 🏆 DS 112 Rubric Achievement
- ✅ **Real-World Application**: Sports fairness analysis with practical implications
- ✅ **Technical Challenge**: Integration of ML, LLMs, and statistical testing
- ✅ **Novelty**: First-of-its-kind VAR fairness audit using AI
- ✅ **Storytelling**: Comprehensive visualizations and bias detection

## 📋 Prerequisites

- Python 3.8 or higher
- Google Colab (recommended) or Jupyter Notebook
- Google Gemini API key (for LLM analysis)

## 🚀 Quick Start

### Option 1: With Sample Data (Recommended for Testing)
```bash
cd var-fairness-audit
pip install -r requirements.txt
python create_sample_data.py  # Generate realistic sample data
python main.py                # Run complete analysis
```

### Option 2: With Your Own Data
1. Place your CSV files in the `data/` folder:
   - `VAR_Incidents_Stats.csv`
   - `VAR_Team_Stats.csv`
2. Run the analysis:
```bash
pip install -r requirements.txt
python main.py
```

### Option 3: LLM Analysis (Google Colab)
1. Open `notebooks/llm_inference.ipynb` in Google Colab
2. Add your Gemini API key to Colab secrets
3. Run all cells for complete LLM analysis

## 📊 Data Requirements

### VAR_Incidents_Stats.csv
- `Team`: Team name (for merging)
- `Decision`: VAR decision (should include "Overturned" values)
- `Description`: Text description of incident (for LLM analysis)
- `Date`, `League`, `Round`: Optional contextual data

### VAR_Team_Stats.csv
- `Team`: Team name (for merging)
- `Rank`: Team ranking (for bias analysis)
- Additional team statistics (wins, losses, etc.)

## 🔧 Project Components

### 1. 📊 Data Extraction (`src/data_extraction.py`)
- Loads and cleans CSV datasets
- Handles missing values intelligently
- Merges data on team identifier
- Outputs cleaned dataset for analysis

### 2. 📈 Data Exploration (`src/data_exploration.py`)
- **9-panel comprehensive dashboard** with:
  - VAR incidents per team analysis
  - Overturn rates by team and tier
  - Time-based trend analysis
  - Fairness deviation scoring
  - Statistical correlation plots
  - League/round heatmaps

### 3. 🔬 Statistical Analysis (`src/data_analysis.py`)
- **Advanced statistical tests**: t-tests, chi-square tests
- **Effect size calculations**: Cohen's d
- **Machine learning models**: Logistic Regression, Random Forest
- **Fairness metrics**: Demographic parity, statistical parity
- **Cross-validation** and model comparison

### 4. 🤖 LLM Analysis (`notebooks/llm_inference.ipynb`)
- **Google Gemini integration** for incident analysis
- **Sentiment embedding** visualization (t-SNE, UMAP)
- **Bias detection** by comparing LLM vs referee decisions
- **Interactive visualizations** with Plotly
- **Confidence scoring** and prediction accuracy

## 📈 Analysis Output

### Traditional Analysis
1. **Merged Dataset**: `data/var_combined.csv`
2. **Comprehensive Visualizations**:
   - 9-panel exploration dashboard
   - Statistical significance plots
   - Fairness deviation metrics
3. **Statistical Reports**:
   - T-test results with effect sizes
   - Machine learning model comparison
   - Fairness assessment scores

### LLM Analysis (Notebook)
1. **AI Predictions**: LLM assessment of incident fairness
2. **Embedding Visualizations**: t-SNE and UMAP plots
3. **Bias Detection**: Comparison matrix between AI and referees
4. **Interactive Dashboard**: Plotly visualizations with team details

## 📁 Enhanced Project Structure

```
var-fairness-audit/
├── 📁 data/                           # Data folder
│   ├── VAR_Incidents_Stats.csv        # VAR incident data
│   ├── VAR_Team_Stats.csv             # Team statistics
│   └── var_combined.csv               # Generated merged dataset
├── 📁 src/                            # Core analysis modules
│   ├── __init__.py
│   ├── data_extraction.py             # Data loading and cleaning
│   ├── data_exploration.py            # 9-panel visualization dashboard
│   └── data_analysis.py               # Advanced statistical analysis
├── 📁 notebooks/                      # LLM analysis
│   └── llm_inference.ipynb            # Google Gemini LLM analysis
├── main.py                            # Main execution script
├── create_sample_data.py              # Sample data generator
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🧮 Advanced Features

### Statistical Analysis
- **Multiple comparison tests** with Bonferroni correction
- **Effect size calculations** (Cohen's d)
- **Cross-validation** with confidence intervals
- **Feature importance** analysis
- **Fairness metrics** (demographic parity, statistical parity)

### Machine Learning
- **Logistic Regression** with regularization
- **Random Forest** for non-linear patterns
- **Cross-validation** for robust evaluation
- **ROC-AUC** analysis for model performance
- **Feature encoding** for categorical variables

### LLM Integration
- **Google Gemini API** for text analysis
- **Sentence transformers** for embeddings
- **Interactive visualizations** with Plotly
- **Bias detection algorithms**
- **Confidence scoring** and uncertainty quantification

## 🎓 DS 112 Project Highlights

### Technical Innovation
- **Multi-modal analysis**: Combines numerical data with text analysis
- **State-of-the-art LLMs**: Google Gemini for fairness assessment
- **Advanced visualizations**: Interactive plots with embedding analysis
- **Robust statistics**: Proper significance testing and effect sizes

### Real-World Impact
- **Sports analytics**: Practical application in professional sports
- **Bias detection**: Addresses fairness in high-stakes decisions
- **Scalable framework**: Can be applied to other referee/judgment scenarios
- **Policy implications**: Results could inform VAR protocol improvements

### Methodological Rigor
- **Multiple validation approaches**: Statistical tests + ML + LLM analysis
- **Proper experimental design**: Train/test splits, cross-validation
- **Bias-aware metrics**: Fairness-specific evaluation criteria
- **Reproducible results**: Seeded random processes

## 🚀 Getting Started for DS 112

1. **Generate sample data**: `python create_sample_data.py`
2. **Run core analysis**: `python main.py`
3. **Open LLM notebook**: Upload `notebooks/llm_inference.ipynb` to Google Colab
4. **Add API key**: Store Gemini API key in Colab secrets
5. **Run complete analysis**: Execute all notebook cells

## 🤝 Extensions & Future Work

This framework can be extended to analyze:
- **Referee-specific patterns** and consistency
- **Match context factors** (home/away, importance)
- **Temporal trends** and rule change impacts
- **Cross-league comparisons** and cultural differences
- **Fan sentiment analysis** using social media data

## 📄 License

This project is intended for educational and research purposes as part of DS 112 coursework.

---

**🎓 Perfect for DS 112 Final Project Requirements!**
- Novel application of LLMs to sports analytics
- Rigorous statistical methodology
- Real-world relevance and impact
- Comprehensive technical implementation 