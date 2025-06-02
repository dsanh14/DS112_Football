#!/usr/bin/env python3
import json
import os

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Basic notebook template
def create_basic_notebook(title, description):
    return {
        "cells": [
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
        ],
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

# Create notebooks
notebooks = [
    {
        "filename": "notebooks/1_Data_Extraction.ipynb",
        "title": "VAR Fairness Audit: Data Extraction",
        "description": "This notebook focuses on loading and preparing the VAR datasets for analysis."
    },
    {
        "filename": "notebooks/2_Data_Exploration.ipynb",
        "title": "VAR Fairness Audit: Data Exploration",
        "description": "This notebook provides visualizations and exploratory analysis of VAR decisions."
    },
    {
        "filename": "notebooks/3_Statistical_Analysis.ipynb",
        "title": "VAR Fairness Audit: Statistical Analysis",
        "description": "This notebook performs statistical tests and ML modeling to detect potential bias."
    },
    {
        "filename": "notebooks/4_LLM_Analysis.ipynb",
        "title": "VAR Fairness Audit: LLM Analysis",
        "description": "This notebook uses Google Gemini to analyze VAR incident descriptions."
    }
]

# Create each notebook
for nb in notebooks:
    notebook = create_basic_notebook(nb["title"], nb["description"])
    with open(nb["filename"], 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {nb['filename']}")

print("\n✅ All notebooks created successfully!")
print("These notebooks can be uploaded to Google Colab and will work out of the box.") 