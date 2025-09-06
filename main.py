import argparse
import pandas as pd
from pipeline import run_full_pipeline

def main():
    """
    Main function to run the NarratorAI pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="NarratorAI: Automated Data Storytelling Bot")
    parser.add_argument("file_path", type=str, help="Path to the CSV file to analyze.")
    parser.add_argument("target_col", type=str, help="Name of the target column for analysis.")
    
    args = parser.parse_args()
    
    try:
        data = pd.read_csv(args.file_path)
        run_full_pipeline(args.file_path, data, args.target_col)
    except FileNotFoundError:
        print(f"Error: The file '{args.file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()