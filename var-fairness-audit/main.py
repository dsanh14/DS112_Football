from src.data_extraction import extract_data
from src.data_exploration import explore_data
from src.data_analysis import analyze_data

def main():
    print("🔍 Starting VAR Fairness Audit...")
    extract_data()
    explore_data()
    analyze_data()

if __name__ == "__main__":
    main() 