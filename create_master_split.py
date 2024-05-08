import argparse
import pandas as pd
from preparation.dataset_functions import create_stratified_splits_master_file, replace_dict

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Generate master split file for BCN_20K dataset.")
    parser.add_argument("--input_csv", type=str, help="Path to the input CSV file (bcn_20k_train.csv)", default= "./bcn_20k_train.csv", required=False)
    parser.add_argument("--output_csv", type=str, help="Path to output the master split CSV file (master_split_file.csv)", default= "./master_split_file.csv", required=False)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load the dataset
    df = pd.read_csv(args.input_csv)

    # Replace the diagnosis column with the label column
    df['label'] = [replace_dict[x] for x in df['diagnosis']]

    # Generate the master split file
    master_df = create_stratified_splits_master_file(df)
    
    # Save the master split file
    master_df.to_csv(args.output_csv, index=False)
    print(f"Master split file has been saved to {args.output_csv}")

if __name__ == "__main__":
    main()

