import json
import csv
import os
import sys

# File paths
base_dir = "input_data"
COMMENTS_FILE = os.path.join(base_dir, 'prod_sunbird__comment_.csv.json')
TREE_FILE = os.path.join(base_dir, 'prod_sunbird_comment_tree.csv.json')

# Output files
OUTPUT_DIR = "comments_data"
DETAILS_CSV = os.path.join(OUTPUT_DIR, 'entity_comments_details.csv')
COUNTS_CSV = os.path.join(OUTPUT_DIR, 'entity_comment_counts.csv')

def main():
    print("Loading comments data... this may take a moment.")
    
    # 1. Load the Comments Data (The Dictionary)
    # We want fast lookup by comment_id
    comments_map = {}
    try:
        with open(COMMENTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # The file structure is { "comment": [ ... ] }
            for item in data.get('comment', []):
                c_id = item.get('comment_id')
                c_data_str = item.get('comment_data')
                
                if not c_id or not c_data_str:
                    continue

                try:
                    c_data = json.loads(c_data_str)
                    comments_map[c_id] = c_data.get('comment', "")
                except json.JSONDecodeError:
                    # Fallback or log error
                    pass
    except FileNotFoundError:
        print(f"Error: Could not find {COMMENTS_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse {COMMENTS_FILE}")
        return

    print(f"Loaded {len(comments_map)} unique comments.")

    # 2. Process the Tree Data (The Map)
    print("Processing entity trees...")
    
    rows_details = []
    rows_counts = []

    try:
        with open(TREE_FILE, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
            
            # The file structure is { "comment_tree": [ ... ] }
            for tree_node in tree_data.get('comment_tree', []):
                # The tree structure is inside a stringified JSON field 'comment_tree_data'
                tree_json_str = tree_node.get('comment_tree_data')
                
                if not tree_json_str:
                    continue

                try:
                    # Parse the nested JSON string
                    tree_details = json.loads(tree_json_str)
                    
                    # Extract entity_id from the parsed inner JSON
                    entity_id = tree_details.get('entityId')
                    
                    if not entity_id:
                        continue
                    
                    # We look for 'childNodes' which usually contains all comments in the thread
                    # If childNodes is missing, we might check 'comments' or 'firstLevelNodes' depending on requirement,
                    # but childNodes is usually the comprehensive list in this schema.
                    child_nodes = tree_details.get('childNodes', [])
                    
                    # If child_nodes is empty, try extracting from 'comments' array if present
                    if not child_nodes and 'comments' in tree_details:
                         child_nodes = [c.get('commentId') for c in tree_details['comments'] if c.get('commentId')]

                    count = 0
                    for comment_id in child_nodes:
                        if not comment_id: 
                            continue
                            
                        # Look up the actual content output
                        comment_text = comments_map.get(comment_id, "")

                        rows_details.append({
                            'entity_id': entity_id,
                            'comment': comment_text
                        })
                        count += 1
                    
                    rows_counts.append({
                        'entity_id': entity_id,
                        'total_comments': count
                    })

                except json.JSONDecodeError:
                    print(f"Warning: Could not parse tree data for a node")
                    continue

    except FileNotFoundError:
        print(f"Error: Could not find {TREE_FILE}")
        return

    # 3. Write Output CSVs
    print(f"Writing {len(rows_details)} rows to {DETAILS_CSV}...")
    with open(DETAILS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['entity_id', 'comment'])
        writer.writeheader()
        writer.writerows(rows_details)

    print(f"Writing counts to {COUNTS_CSV}...")
    with open(COUNTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['entity_id', 'total_comments'])
        writer.writeheader()
        writer.writerows(rows_counts)

    print("Done!")

if __name__ == "__main__":
    main()
