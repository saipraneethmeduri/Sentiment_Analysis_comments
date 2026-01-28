import json

COMMENTS_FILE = 'prod_sunbird__comment_.csv.json'
TREE_FILE = 'prod_sunbird_comment_tree.csv.json'

def debug():
    # 1. Load a few comments
    print("Loading small subset of comments...")
    comments_sample = set()
    with open(COMMENTS_FILE, 'r') as f:
        data = json.load(f)
        for i, item in enumerate(data.get('comment', [])):
            comments_sample.add(item.get('comment_id'))
            if i < 3:
                print(f"Sample Comment ID: {item.get('comment_id')}")
    
    print(f"Total Comments in list: {len(data.get('comment', []))}")

    # 2. Inspect Tree
    print("\nInspecting Tree Data...")
    with open(TREE_FILE, 'r') as f:
        tree_data = json.load(f)
        trees = tree_data.get('comment_tree', [])
        print(f"Total Trees: {len(trees)}")
        
        if len(trees) > 0:
            first_tree = trees[0]
            print(f"First Tree Keys: {first_tree.keys()}")
            
            raw_data = first_tree.get('comment_tree_data')
            print(f"Raw Data Type: {type(raw_data)}")
            
            if raw_data:
                try:
                    parsed = json.loads(raw_data)
                    print(f"Parsed Keys: {parsed.keys()}")
                    nodes = parsed.get('childNodes', [])
                    print(f"Found {len(nodes)} childNodes.")
                    if len(nodes) > 0:
                        print(f"Sample Node ID: {nodes[0]}")
                        if nodes[0] in comments_sample:
                            print("MATCH FOUND: Node ID exists in comments file.")
                        else:
                            print("NO MATCH: Node ID NOT found in comments file (checked full file).")
                except Exception as e:
                    print(f"JSON Parse Error: {e}")

debug()
