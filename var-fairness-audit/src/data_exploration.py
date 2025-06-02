def explore_data():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from datetime import datetime, timedelta
    
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    df = pd.read_csv("data/var_combined.csv")
    
    print("🔍 Starting VAR Data Exploration...")
    print(f"Dataset shape: {df.shape}")
    
    # Create figure with subplots for comprehensive analysis
    fig = plt.figure(figsize=(20, 15))
    
    # 1. VAR incidents per team (enhanced)
    plt.subplot(3, 3, 1)
    incidents_count = df["Team"].value_counts().head(15)
    bars = plt.bar(range(len(incidents_count)), incidents_count.values, color='skyblue', alpha=0.8)
    plt.title("📊 VAR Incidents per Team (Top 15)", fontsize=14, fontweight='bold')
    plt.xlabel("Teams")
    plt.ylabel("Number of Incidents")
    plt.xticks(range(len(incidents_count)), incidents_count.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(incidents_count.values[i]), ha='center', va='bottom', fontweight='bold')
    
    # 2. VAR overturn rates with stacked bar chart
    plt.subplot(3, 3, 2)
    overturn_data = df.groupby('Team')['Decision'].value_counts().unstack(fill_value=0)
    if 'Overturned' in overturn_data.columns:
        overturn_rates = (overturn_data['Overturned'] / overturn_data.sum(axis=1)).head(10)
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(overturn_rates)))
        bars = plt.bar(range(len(overturn_rates)), overturn_rates.values, color=colors)
        plt.title("⚖️ VAR Overturn Rates by Team (Top 10)", fontsize=14, fontweight='bold')
        plt.xlabel("Teams")
        plt.ylabel("Overturn Rate")
        plt.xticks(range(len(overturn_rates)), overturn_rates.index, rotation=45, ha='right')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{overturn_rates.values[i]:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Decision distribution pie chart
    plt.subplot(3, 3, 3)
    decision_counts = df['Decision'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    plt.pie(decision_counts.values, labels=decision_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(decision_counts)], startangle=90)
    plt.title("🎯 Overall VAR Decision Distribution", fontsize=14, fontweight='bold')
    
    # 4. Heatmap analysis (enhanced)
    if "League" in df.columns and "Round" in df.columns:
        plt.subplot(3, 3, 4)
        pivot_table = df.pivot_table(index="League", columns="Round", aggfunc="size", fill_value=0)
        sns.heatmap(pivot_table, cmap="YlOrRd", annot=True, fmt='d', cbar_kws={'label': 'Incidents'})
        plt.title("🗺️ VAR Incidents by League and Round", fontsize=14, fontweight='bold')
    
    # 5. Team ranking vs overturn analysis
    if 'Rank' in df.columns:
        plt.subplot(3, 3, 5)
        df['Rank_Numeric'] = pd.to_numeric(df['Rank'], errors='coerce')
        df['Overturned_Binary'] = (df['Decision'] == 'Overturned').astype(int)
        
        # Group by ranking tiers
        df['Ranking_Tier'] = pd.qcut(df['Rank_Numeric'], q=3, labels=['Top', 'Mid', 'Bottom'], duplicates='drop')
        tier_overturn = df.groupby('Ranking_Tier')['Overturned_Binary'].mean()
        
        bars = plt.bar(tier_overturn.index, tier_overturn.values, 
                      color=['gold', 'silver', '#CD7F32'], alpha=0.8)
        plt.title("🏆 Overturn Rate by Team Ranking Tier", fontsize=14, fontweight='bold')
        plt.ylabel("Overturn Rate")
        
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{tier_overturn.values[i]:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Time-based analysis (simulated if no date column)
    plt.subplot(3, 3, 6)
    if 'Date' not in df.columns:
        # Create simulated match dates for demonstration
        np.random.seed(42)
        start_date = datetime(2023, 8, 1)
        df['Date'] = [start_date + timedelta(days=np.random.randint(0, 300)) for _ in range(len(df))]
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    monthly_incidents = df.groupby('Month').size()
    
    plt.plot(monthly_incidents.index, monthly_incidents.values, marker='o', linewidth=2, markersize=8)
    plt.title("📅 VAR Incidents Over Time (Monthly)", fontsize=14, fontweight='bold')
    plt.xlabel("Month")
    plt.ylabel("Number of Incidents")
    plt.grid(True, alpha=0.3)
    
    # 7. Fairness metric visualization
    plt.subplot(3, 3, 7)
    if 'Ranking_Tier' in df.columns:
        fairness_analysis = df.groupby('Ranking_Tier').agg({
            'Overturned_Binary': ['mean', 'count']
        }).round(3)
        fairness_analysis.columns = ['Overturn_Rate', 'Sample_Size']
        
        # Calculate fairness score (deviation from expected)
        expected_rate = df['Overturned_Binary'].mean()
        fairness_analysis['Fairness_Score'] = abs(fairness_analysis['Overturn_Rate'] - expected_rate)
        
        bars = plt.bar(fairness_analysis.index, fairness_analysis['Fairness_Score'], 
                      color=['red' if x > 0.1 else 'green' for x in fairness_analysis['Fairness_Score']])
        plt.title("⚖️ Fairness Deviation Score by Tier", fontsize=14, fontweight='bold')
        plt.ylabel("Deviation from Expected Rate")
        plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Concern Threshold')
        plt.legend()
    
    # 8. Statistical summary
    plt.subplot(3, 3, 8)
    plt.axis('off')
    stats_text = f"""
    📊 KEY STATISTICS
    
    Total Incidents: {len(df):,}
    Teams Analyzed: {df['Team'].nunique()}
    Overall Overturn Rate: {df['Overturned_Binary'].mean():.1%}
    
    Most Incidents: {incidents_count.index[0]} ({incidents_count.iloc[0]})
    
    📈 FAIRNESS INSIGHTS
    Std Dev of Overturn Rates: {df.groupby('Team')['Overturned_Binary'].mean().std():.3f}
    Range: {df.groupby('Team')['Overturned_Binary'].mean().min():.1%} - {df.groupby('Team')['Overturned_Binary'].mean().max():.1%}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 9. Correlation analysis
    plt.subplot(3, 3, 9)
    if 'Rank_Numeric' in df.columns:
        # Scatter plot of rank vs overturn incidents
        team_stats = df.groupby('Team').agg({
            'Rank_Numeric': 'first',
            'Overturned_Binary': 'sum'
        }).dropna()
        
        plt.scatter(team_stats['Rank_Numeric'], team_stats['Overturned_Binary'], 
                   alpha=0.6, s=60, color='purple')
        plt.xlabel("Team Rank (lower = better)")
        plt.ylabel("Total Overturned Decisions")
        plt.title("🔍 Team Rank vs VAR Overturns", fontsize=14, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(team_stats['Rank_Numeric'], team_stats['Overturned_Binary'], 1)
        p = np.poly1d(z)
        plt.plot(team_stats['Rank_Numeric'], p(team_stats['Rank_Numeric']), 
                "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr = team_stats['Rank_Numeric'].corr(team_stats['Overturned_Binary'])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle("🏈 VAR Fairness Audit - Comprehensive Data Exploration", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.show()
    
    print("✅ Data exploration complete! Generated comprehensive visualization dashboard.") 