def analyze_data():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')

    df = pd.read_csv("data/var_combined.csv")
    
    print("=" * 60)
    print("🔬 VAR FAIRNESS AUDIT - ADVANCED STATISTICAL ANALYSIS")
    print("=" * 60)

    # Data preparation
    df["Overturned"] = df["Decision"].apply(lambda x: 1 if x == "Overturned" else 0)
    df["Rank"] = pd.to_numeric(df["Rank"], errors='coerce')
    
    # Summary statistics
    print("\n📊 COMPREHENSIVE SUMMARY STATISTICS:")
    print("-" * 40)
    
    total_incidents = len(df)
    total_teams = df["Team"].nunique()
    overall_overturn_rate = df["Overturned"].mean()
    
    print(f"Total VAR Incidents: {total_incidents:,}")
    print(f"Teams Analyzed: {total_teams}")
    print(f"Overall Overturn Rate: {overall_overturn_rate:.1%}")
    print(f"Standard Deviation: {df['Overturned'].std():.3f}")
    
    # Team-level statistics
    team_stats = df.groupby("Team").agg({
        'Overturned': ['count', 'sum', 'mean'],
        'Rank': 'first'
    }).round(3)
    team_stats.columns = ['Total_Incidents', 'Overturned_Count', 'Overturn_Rate', 'Team_Rank']
    team_stats = team_stats.sort_values('Overturn_Rate', ascending=False)
    
    print(f"\n🏆 TOP 5 TEAMS BY OVERTURN RATE:")
    print(team_stats.head().to_string())
    
    print(f"\n🛡️ BOTTOM 5 TEAMS BY OVERTURN RATE:")
    print(team_stats.tail().to_string())

    # Advanced bias analysis with statistical tests
    print("\n⚖️ ADVANCED BIAS DETECTION ANALYSIS:")
    print("-" * 40)
    
    # Create ranking tiers for comparison
    df_clean = df.dropna(subset=['Rank'])
    if len(df_clean) > 0:
        df_clean["Tier"] = pd.qcut(df_clean["Rank"], q=3, labels=["Top", "Mid", "Bottom"], duplicates='drop')
        
        # Group-wise overturn rates
        tier_analysis = df_clean.groupby("Tier")["Overturned"].agg(['count', 'mean', 'std']).round(4)
        tier_analysis.columns = ['Sample_Size', 'Overturn_Rate', 'Std_Dev']
        print("\n📈 Overturn Analysis by Team Tier:")
        print(tier_analysis.to_string())
        
        # Statistical significance tests
        print("\n🧪 STATISTICAL SIGNIFICANCE TESTS:")
        print("-" * 30)
        
        # Two-sample t-test: Top vs Bottom tier
        if 'Top' in df_clean['Tier'].values and 'Bottom' in df_clean['Tier'].values:
            top_tier = df_clean[df_clean["Tier"] == "Top"]["Overturned"]
            bottom_tier = df_clean[df_clean["Tier"] == "Bottom"]["Overturned"]
            
            t_stat, p_value = stats.ttest_ind(top_tier, bottom_tier)
            print(f"Top vs Bottom Tier T-test:")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(top_tier)-1)*top_tier.std()**2 + (len(bottom_tier)-1)*bottom_tier.std()**2) / (len(top_tier)+len(bottom_tier)-2))
            cohens_d = (top_tier.mean() - bottom_tier.mean()) / pooled_std
            print(f"  Effect Size (Cohen's d): {cohens_d:.4f}")
            
            effect_interpretation = "Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large"
            print(f"  Effect Size Interpretation: {effect_interpretation}")
        
        # Chi-square test for independence
        if len(df_clean['Tier'].unique()) > 1:
            contingency_table = pd.crosstab(df_clean['Tier'], df_clean['Overturned'])
            chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
            print(f"\nChi-square Test of Independence:")
            print(f"  Chi-square statistic: {chi2:.4f}")
            print(f"  P-value: {p_chi2:.4f}")
            print(f"  Degrees of freedom: {dof}")
            print(f"  Significant: {'Yes' if p_chi2 < 0.05 else 'No'} (α = 0.05)")

    # Advanced Machine Learning Analysis
    print("\n🤖 ADVANCED MACHINE LEARNING ANALYSIS:")
    print("-" * 40)
    
    # Prepare features for ML
    ml_features = []
    feature_names = []
    
    # Add ranking feature
    if 'Rank' in df.columns:
        rank_clean = pd.to_numeric(df['Rank'], errors='coerce')
        if not rank_clean.isna().all():
            ml_features.append(rank_clean.fillna(rank_clean.median()))
            feature_names.append('Team_Rank')
    
    # Add team encoding
    if 'Team' in df.columns:
        le_team = LabelEncoder()
        team_encoded = le_team.fit_transform(df['Team'])
        ml_features.append(team_encoded)
        feature_names.append('Team_Encoded')
    
    # Add league encoding if available
    if 'League' in df.columns:
        le_league = LabelEncoder()
        league_encoded = le_league.fit_transform(df['League'].fillna('Unknown'))
        ml_features.append(league_encoded)
        feature_names.append('League_Encoded')
    
    # Create feature matrix
    if ml_features:
        X = np.column_stack(ml_features)
        y = df['Overturned'].values
        
        # Remove rows with missing target
        valid_indices = ~pd.isna(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) > 10:  # Ensure sufficient data
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 1. Logistic Regression (Enhanced)
            print("\n📊 Logistic Regression Results:")
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train_scaled, y_train)
            
            y_pred_lr = lr_model.predict(X_test_scaled)
            y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
            
            print(f"  Accuracy: {lr_model.score(X_test_scaled, y_test):.3f}")
            try:
                auc_lr = roc_auc_score(y_test, y_proba_lr)
                print(f"  AUC-ROC: {auc_lr:.3f}")
            except:
                print("  AUC-ROC: Not available (single class)")
            
            # Cross-validation
            cv_scores_lr = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
            print(f"  CV Accuracy: {cv_scores_lr.mean():.3f} (±{cv_scores_lr.std()*2:.3f})")
            
            # Feature importance
            if len(feature_names) == len(lr_model.coef_[0]):
                print("\n  Feature Importance (Coefficients):")
                for name, coef in zip(feature_names, lr_model.coef_[0]):
                    print(f"    {name}: {coef:.4f}")
            
            # 2. Random Forest (for comparison)
            print("\n🌲 Random Forest Results:")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            y_pred_rf = rf_model.predict(X_test)
            print(f"  Accuracy: {rf_model.score(X_test, y_test):.3f}")
            
            try:
                y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
                auc_rf = roc_auc_score(y_test, y_proba_rf)
                print(f"  AUC-ROC: {auc_rf:.3f}")
            except:
                print("  AUC-ROC: Not available (single class)")
            
            cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5)
            print(f"  CV Accuracy: {cv_scores_rf.mean():.3f} (±{cv_scores_rf.std()*2:.3f})")
            
            # Feature importance for Random Forest
            if len(feature_names) == len(rf_model.feature_importances_):
                print("\n  Feature Importance (Gini):")
                feature_importance = sorted(zip(feature_names, rf_model.feature_importances_), 
                                          key=lambda x: x[1], reverse=True)
                for name, importance in feature_importance:
                    print(f"    {name}: {importance:.4f}")
            
            # Model comparison
            print(f"\n🏆 Model Comparison:")
            print(f"  Logistic Regression CV: {cv_scores_lr.mean():.3f}")
            print(f"  Random Forest CV: {cv_scores_rf.mean():.3f}")
            best_model = "Random Forest" if cv_scores_rf.mean() > cv_scores_lr.mean() else "Logistic Regression"
            print(f"  Best Model: {best_model}")

    # Fairness metrics calculation
    print("\n⚖️ FAIRNESS METRICS SUMMARY:")
    print("-" * 30)
    
    if 'Tier' in df_clean.columns:
        # Calculate fairness metrics
        overall_rate = df_clean['Overturned'].mean()
        
        fairness_metrics = {}
        for tier in df_clean['Tier'].unique():
            tier_data = df_clean[df_clean['Tier'] == tier]
            tier_rate = tier_data['Overturned'].mean()
            
            # Demographic parity difference
            dp_diff = abs(tier_rate - overall_rate)
            
            # Statistical parity
            stat_parity = tier_rate / overall_rate if overall_rate > 0 else 1
            
            fairness_metrics[tier] = {
                'Overturn_Rate': tier_rate,
                'DP_Difference': dp_diff,
                'Stat_Parity_Ratio': stat_parity,
                'Sample_Size': len(tier_data)
            }
        
        fairness_df = pd.DataFrame(fairness_metrics).T.round(4)
        print(fairness_df.to_string())
        
        # Fairness assessment
        max_dp_diff = fairness_df['DP_Difference'].max()
        print(f"\nMaximum Demographic Parity Difference: {max_dp_diff:.4f}")
        
        if max_dp_diff < 0.05:
            fairness_assessment = "✅ HIGH FAIRNESS"
        elif max_dp_diff < 0.1:
            fairness_assessment = "⚠️ MODERATE CONCERNS"
        else:
            fairness_assessment = "❌ SIGNIFICANT BIAS DETECTED"
        
        print(f"Fairness Assessment: {fairness_assessment}")

    # Final recommendations
    print("\n💡 KEY INSIGHTS & RECOMMENDATIONS:")
    print("-" * 40)
    
    insights = []
    
    if 'tier_analysis' in locals():
        highest_tier = tier_analysis['Overturn_Rate'].idxmax()
        lowest_tier = tier_analysis['Overturn_Rate'].idxmin()
        rate_diff = tier_analysis.loc[highest_tier, 'Overturn_Rate'] - tier_analysis.loc[lowest_tier, 'Overturn_Rate']
        
        insights.append(f"• {highest_tier} tier teams have {rate_diff:.1%} higher overturn rate than {lowest_tier} tier")
    
    if 'p_value' in locals() and p_value < 0.05:
        insights.append(f"• Statistically significant difference found between team tiers (p={p_value:.4f})")
    
    if 'max_dp_diff' in locals():
        if max_dp_diff > 0.1:
            insights.append(f"• Significant fairness concerns detected (max difference: {max_dp_diff:.1%})")
        else:
            insights.append(f"• Fairness levels within acceptable range (max difference: {max_dp_diff:.1%})")
    
    if insights:
        for insight in insights:
            print(insight)
    else:
        print("• Analysis complete - review results above for detailed insights")
    
    print("\n" + "=" * 60)
    print("✅ ADVANCED STATISTICAL ANALYSIS COMPLETE")
    print("=" * 60) 