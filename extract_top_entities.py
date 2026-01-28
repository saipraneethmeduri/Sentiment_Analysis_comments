import pandas as pd
import os

def main():
    # Define file paths
    base_dir = "comments_data"
    counts_file = os.path.join(base_dir, "entity_comment_counts.csv")
    details_file = os.path.join(base_dir, "entity_comments_details.csv")
    
    output_counts_file = os.path.join(base_dir, "entity_comment_counts_20.csv")
    output_details_file = os.path.join(base_dir, "entity_comments_details_20.csv")

    print("Loading data...")
    # Load the counts data
    try:
        df_counts = pd.read_csv(counts_file)
        print(f"Loaded {len(df_counts)} rows from {counts_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {counts_file}")
        return

    # Sort by total_comments descending and take top 20
    # Take first 20 rows (no sorting)
    # Note: adjusting column name based on file inspection if needed, 
    # but based on previous view_file, columns are 'entity_id' and 'total_comments'
    if 'total_comments' not in df_counts.columns:
        print(f"Error: 'total_comments' column not found. Columns: {df_counts.columns}")
        return

    df_top_20 = df_counts.head(20)
    
    # Save top 20 counts
    df_top_20.to_csv(output_counts_file, index=False)
    print(f"Saved top 20 counts to {output_counts_file}")

    # Get the list of top 20 entity_ids
    top_20_ids = df_top_20['entity_id'].tolist()

    print("Loading details data...")
    # Load the details data
    # iterating or loading fully depending on size? 43MB is small enough to load fully.
    try:
        df_details = pd.read_csv(details_file)
        print(f"Loaded {len(df_details)} rows from {details_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {details_file}")
        return

    # Filter for top 20 entities
    df_details_20 = df_details[df_details['entity_id'].isin(top_20_ids)]
    
    # Save filtered details
    df_details_20.to_csv(output_details_file, index=False)
    print(f"Saved details for top 20 entities to {output_details_file}")
    print(f"Count of comments saved: {len(df_details_20)}")

if __name__ == "__main__":
    main()
