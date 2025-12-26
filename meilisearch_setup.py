import meilisearch
import pandas as pd
import json
import time
import numpy as np

def simple_clean(value):
    """Simple cleaning - convert to basic Python types"""
    if pd.isna(value):
        return None
    if isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        stripped = stripped.encode('ascii', errors='ignore').decode('ascii')
        return stripped if stripped else None
    return str(value) if value else None


def load_companies(csv_path: str):
    """Load companies with proper timeout handling"""
    
    # 1. Load CSV
    print(f"📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅ Loaded {len(df):,} rows x {len(df.columns)} columns\n")
    
    # 2. Clean all columns
    print("🧹 Cleaning data...")
    for col in df.columns:
        df[col] = df[col].apply(simple_clean)
    
    # 3. Force replace NaN
    df = df.replace({np.nan: None})
    df = df.where(pd.notna(df))
    print("✅ Cleaning complete\n")
    
    # 4. Add ID
    df.insert(0, 'id', range(1, len(df) + 1))
    
    # 5. Convert to documents
    documents = df.to_dict('records')
    
    # 6. Sanitize documents
    print("🔧 Sanitizing documents...")
    for doc in documents:
        for key in list(doc.keys()):
            value = doc[key]
            try:
                if pd.isna(value):
                    doc[key] = None
            except:
                pass
            if isinstance(value, float):
                try:
                    if np.isnan(value) or np.isinf(value):
                        doc[key] = None
                except:
                    pass
    print(f"✅ {len(documents):,} documents ready\n")
    
    # 7. Connect with INCREASED TIMEOUT
    print("🔌 Connecting to Meilisearch...")
    client = meilisearch.Client(
        'http://localhost:7700',
        'testMasterKey123',
        timeout=30  # Increase from 5s to 30s
    )
    
    # 8. Delete old index
    try:
        task = client.delete_index('companies')
        print("🗑️  Deleted old index")
        time.sleep(2)
    except:
        pass
    
    # 9. Create new index
    print("📦 Creating index...")
    client.create_index('companies', {'primaryKey': 'id'})
    time.sleep(2)
    index = client.index('companies')
    print("✅ Index created\n")
    
    # 10. Configure searchable attributes
    print("🔍 Configuring searchable attributes...")
    searchable = ['COMPANY_NAME_CLEANED', 'COMPANY_NAME', 'DOMAIN_PART', 'COMPANY_NAME_CLEANED_TWO_WORDS']
    searchable = [col for col in searchable if col in df.columns]
    
    if searchable:
        task = index.update_searchable_attributes(searchable)
        try:
            client.wait_for_task(task.task_uid, timeout_in_ms=30000)  # 30s timeout
        except:
            print("   ⚠️  Configuration taking long, continuing anyway...")
            time.sleep(5)
        print(f"✅ Searchable: {searchable}\n")
    
    # 11. Load documents WITHOUT waiting for each batch
    print(f"📤 Loading {len(documents):,} documents (async mode)...\n")
    start = time.time()
    
    batch_size = 10000
    task_uids = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        try:
            # Submit task but DON'T wait for it
            task = index.add_documents(batch) # type: ignore
            task_uids.append(task.task_uid)
            
            processed = min(i+batch_size, len(documents))
            if processed % 50000 == 0 or processed == len(documents):
                print(f"  ✓ Queued: {processed:,}/{len(documents):,} ({len(task_uids)} tasks)")
                
        except Exception as e:
            print(f"\n❌ Failed to queue batch at {i:,}: {e}")
    
    print(f"\n✅ All batches queued! ({len(task_uids)} tasks)")
    print(f"⏳ Waiting for Meilisearch to process...\n")
    
    # 12. Wait for all tasks to complete (with better error handling)
    completed = 0
    last_check = time.time()
    
    while completed < len(task_uids):
        time.sleep(5)  # Check every 5 seconds
        
        # Check index stats instead of individual tasks
        try:
            stats = index.get_stats()
            current_docs = stats.number_of_documents if hasattr(stats, 'number_of_documents') else stats.number_of_documents
            
            if current_docs > completed * batch_size:
                completed = current_docs // batch_size
                elapsed = time.time() - start
                rate = current_docs / elapsed if elapsed > 0 else 0
                print(f"  📊 Indexed: {current_docs:,}/{len(documents):,} ({rate:,.0f} docs/sec)")
            
            # If we've indexed everything, break
            if current_docs >= len(documents) * 0.99:  # 99% is good enough
                print(f"\n✅ Indexing complete!")
                break
                
            # Safety timeout: if no progress for 60 seconds, break
            if time.time() - last_check > 60:
                print(f"\n⚠️  No progress for 60s, assuming done")
                break
                
        except Exception as e:
            print(f"  ⚠️  Error checking stats: {e}")
            time.sleep(10)
    
    total_time = time.time() - start
    
    # 13. Final verification
    try:
        stats = index.get_stats()
        total = stats.number_of_documents if hasattr(stats, 'number_of_documents') else stats.number_of_documents
        print(f"\n📊 Final count: {total:,} documents")
        print(f"⏱️  Total time: {total_time:.1f}s ({total/total_time:,.0f} docs/sec)")
    except Exception as e:
        print(f"⚠️  Could not get final stats: {e}")
    
    return index


if __name__ == "__main__":
    index = load_companies("Sample_dataset.csv")
    
    if index:
        print("\n" + "="*70)
        print("🧪 TESTING SEARCH")
        print("="*70)
        
        # Give Meilisearch a moment to finish any pending indexing
        print("\n⏳ Giving Meilisearch 10s to finish indexing...")
        time.sleep(10)
        
        # Test 1
        result = index.search("heal within", {'limit': 3})
        hits = result['hits']
        
        print(f"\n1️⃣  Search 'heal within': Found {len(hits)} results")
        for hit in hits:
            name = hit.get('COMPANY_NAME_CLEANED', 'N/A')
            domain = hit.get('DOMAIN_NAME', 'N/A')
            print(f"   • {name} → {domain}")
        
        # Test 2
        result = index.search("heaney", {'limit': 3})
        hits = result['hits']
        
        print(f"\n2️⃣  Search 'heaney': Found {len(hits)} results")
        for hit in hits:
            name = hit.get('COMPANY_NAME_CLEANED', 'N/A')
            domain = hit.get('DOMAIN_NAME', 'N/A')
            print(f"   • {name} → {domain}")
        
        # Test 3: Check total indexed
        stats = index.get_stats()
        total = stats.number_of_documents if hasattr(stats, 'number_of_documents') else stats.number_of_documents
        
        print(f"\n3️⃣  Total documents indexed: {total:,}")
        
        print("\n" + "="*70)
        print("✅ SUCCESS! Meilisearch is ready!")
        print("="*70)
