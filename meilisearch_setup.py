import meilisearch
import pandas as pd
import json
import time
import numpy as np
import os

def simple_clean(value):
    """Simple cleaning - convert to basic Python types (used for CSV only)"""
    if pd.isna(value):
        return None
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        if pd.isna(value): return None
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped: return None
        try:
            stripped = stripped.encode('ascii', errors='ignore').decode('ascii')
        except:
            pass
        return stripped if stripped else None
    return str(value) if value else None

def load_data_file(file_path: str):
    """
    Intelligently load either CSV or JSON file into a list of dictionaries (documents).
    """
    print(f"📂 Loading {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = file_path.lower().split('.')[-1]

    if file_ext == 'json':
        # --- JSON PATH ---
        print("   Detected JSON format.")
        with open(file_path, 'r') as f:
            documents = json.load(f)
        
        # Ensure IDs exist
        print("   Validating IDs...")
        for i, doc in enumerate(documents):
            if 'id' not in doc:
                doc['id'] = i + 1
        
        print(f"✅ Loaded {len(documents):,} documents from JSON")
        return documents

    elif file_ext == 'csv':
        # --- CSV PATH ---
        print("   Detected CSV format.")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} rows x {len(df.columns)} columns")

        # 2. Clean
        print("🧹 Cleaning CSV data...")
        for col in df.columns:
            df[col] = df[col].apply(simple_clean)
        
        # 3. Handle NaNs
        df = df.replace({np.nan: None})
        df = df.where(pd.notna(df))
        
        # 4. Add ID
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1, len(df) + 1))
        
        documents = df.to_dict('records')
        
        # 5. Sanitize
        print("🔧 Sanitizing documents...")
        for doc in documents:
            for key in list(doc.keys()):
                value = doc[key]
                # Double check for lingering NaNs
                if value is None: continue
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    doc[key] = None
        
        print(f"✅ Converted to {len(documents):,} documents")
        return documents

    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please use .csv or .json")


def setup_meilisearch(file_path: str):
    """Main function to setup Meilisearch index from file"""
    
    # 1. Load Data (CSV or JSON)
    documents = load_data_file(file_path)
    
    # 2. Connect
    print("\n🔌 Connecting to Meilisearch...")
    client = meilisearch.Client('http://localhost:7700', 'testMasterKey123', timeout=30)
    
    # 3. Reset Index
    index_name = 'companies'
    try:
        client.delete_index(index_name)
        print("🗑️  Deleted old index")
        time.sleep(1)
    except:
        pass
    
    print(f"📦 Creating index '{index_name}'...")
    client.create_index(index_name, {'primaryKey': 'id'})
    time.sleep(1)
    index = client.index(index_name)
    
    # 4. Configure Settings (Important for new columns!)
    print("🔍 Configuring searchable attributes...")
    
    # Detect available columns from first document
    sample_doc = documents[0] if documents else {}
    potential_fields = [
        'company_name_cleaned_ascii', 'COMPANY_NAME_CLEANED', 
        'COMPANY_NAME', 'DOMAIN_PART', 
        'domain_parts_ngram',    # New from preprocessing
        'domain_phonetic',       # New from preprocessing
        'company_phonetic',      # New from preprocessing
        'alternative_names'      # New from preprocessing
    ]
    
    searchable = [field for field in potential_fields if field in sample_doc]
    
    if searchable:
        task = index.update_searchable_attributes(searchable)
        print(f"   Searchable attributes set: {searchable}")
    
    # Configure Ranking Rules (Metadata boosting)
    print("🏆 Configuring ranking rules...")
    ranking_rules = [
        'words', 'typo', 'proximity', 'attribute', 'sort', 'exactness',
        'metadata_quality_score:desc', # New from preprocessing
        'EMPLOYEE_COUNT:desc'
    ]
    index.update_ranking_rules(ranking_rules)
    
    # 5. Load Documents (Async Batching)
    print(f"\n📤 Loading {len(documents):,} documents...")
    
    batch_size = 10000
    task_uids = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        try:
            task = index.add_documents(batch) # type: ignore
            task_uids.append(task.task_uid)
            if i % 50000 == 0:
                print(f"   Queued batch {i // batch_size + 1}...")
        except Exception as e:
            print(f"❌ Error queuing batch {i}: {e}")

    # 6. Wait for Completion
    print(f"⏳ Waiting for processing ({len(task_uids)} batches)...")
    
    while True:
        stats = index.get_stats()
        count = stats.number_of_documents
        if count >= len(documents) * 0.99:
            print(f"\n✅ Indexing Complete! Total docs: {count:,}")
            break
        print(f"   Indexed: {count:,} / {len(documents):,}")
        time.sleep(2)
        
    return index

if __name__ == "__main__":
    # Example usage: Can now pass CSV or JSON
    # index = setup_meilisearch("companies_1M.csv") 
    index = setup_meilisearch("companies_processed.json")
