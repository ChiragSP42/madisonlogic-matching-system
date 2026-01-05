import meilisearch
import pandas as pd
import numpy as np
import boto3
import io
import re
import time
import uuid

# --- CONFIGURATION ---
MEILI_URL = 'http://localhost:7700'
MEILI_KEY = 'testMasterKey123'
INDEX_NAME = 'companies'

# S3 Config
AWS_BUCKET = 'ml-predictiff-data-share'
AWS_PREFIX = 'database/' # Folder containing parquet files
AWS_REGION = 'us-east-1'

# Batching Config
# Meilisearch handles ~5MB-10MB payloads best.
# 51M records is huge, so we process in reasonably sized chunks.
UPLOAD_BATCH_SIZE = 5000 

# --- 1. PREPROCESSING LOGIC (Preserved from your script) ---

def clean_text_basic(text):
    """Standardizes text: lowercase, ASCII only."""
    if not isinstance(text, str): return ""
    try:
        text = text.encode('ascii', errors='ignore').decode('ascii')
    except:
        pass
    return text.lower().strip()

def get_ngrams(text, min_len=3, max_len=15):
    """Generates n-grams for partial matching."""
    if not text or not isinstance(text, str): return []
    text = clean_text_basic(text)
    if len(text) < min_len: return [text]
    return [text[:i] for i in range(min_len, min(len(text) + 1, max_len + 1))]

def simple_phonetic(text):
    """Generates phonetic code."""
    if not text or not isinstance(text, str): return ""
    text = clean_text_basic(text).upper()
    if not text: return ""
    first = text[0]
    remainder = re.sub(r'[AEIOUY]', '', text[1:])
    remainder = re.sub(r'[BFPV]', '1', remainder)
    remainder = re.sub(r'[CGJKQSXZ]', '2', remainder)
    remainder = re.sub(r'[DT]', '3', remainder)
    remainder = re.sub(r'[L]', '4', remainder)
    remainder = re.sub(r'[MN]', '5', remainder)
    remainder = re.sub(r'[R]', '6', remainder)
    
    reduced = ""
    last_char = ""
    for char in remainder:
        if char != last_char:
            reduced += char
        last_char = char
    return first + reduced

def calc_quality_score(row):
    """Calculates Metadata Quality Score."""
    score = 0
    src = str(row.get('SOURCE', '')).upper()
    if 'PDL' in src: score += 20
    elif 'BOMBORA' in src: score += 15
    elif 'HGDATA' in src: score += 10
    
    if pd.notna(row.get('EMPLOYEE_COUNT')) and row.get('EMPLOYEE_COUNT') > 0: score += 10
    if pd.notna(row.get('INDUSTRY_CAT_STD')): score += 5
    if pd.notna(row.get('COUNTRY')): score += 2
    if pd.notna(row.get('SIZE_DESC_STD')): score += 3
    if pd.notna(row.get('LAST_SEEN_DATE')): score += 5
    return score

def process_dataframe(df):
    """Applies all cleaning logic to a Pandas DataFrame."""
    
    # 1. Ensure ID exists. With 51M records, simple counters are risky across files.
    # If your data has a UNIQUE ID in snowflake, use it. 
    # Otherwise, generate a deterministic hash or UUID.
    if 'id' not in df.columns:
        # Generating UUIDs to be safe across disjointed parquet files
        df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]

    # 2. Text Normalization
    df['company_name_cleaned_ascii'] = df['COMPANY_NAME_CLEANED'].apply(clean_text_basic)
    
    # 3. N-grams
    if 'DOMAIN_PART' not in df.columns:
         df['DOMAIN_PART'] = df['DOMAIN_NAME'].astype(str).apply(lambda x: x.split('.')[0] if '.' in x else x)
    df['domain_parts_ngram'] = df['DOMAIN_PART'].apply(get_ngrams)
    
    # 4. Phonetics
    df['domain_phonetic'] = df['DOMAIN_PART'].apply(simple_phonetic)
    df['company_phonetic'] = df['COMPANY_NAME_CLEANED'].apply(simple_phonetic)
    
    # 5. Ranking Scores
    df['metadata_quality_score'] = df.apply(calc_quality_score, axis=1)
    
    # 6. Clean for JSON serialization (Handle NaN/Infinite)
    df = df.replace({np.nan: None})
    records = df.to_dict(orient='records')
    
    # Remove None keys to save network bandwidth
    cleaned_records = []
    for r in records:
        clean_r = {k: v for k, v in r.items() if v is not None and v != ""}
        cleaned_records.append(clean_r)
        
    return cleaned_records

# --- 2. MEILISEARCH SETUP ---

def init_meilisearch():
    print(f"🔌 Connecting to Meilisearch at {MEILI_URL}...")
    client = meilisearch.Client(MEILI_URL, MEILI_KEY, timeout=None) # Disable timeout for large jobs
    
    # Check if index exists, if not create and configure
    try:
        client.get_index(INDEX_NAME)
        print(f"   Index '{INDEX_NAME}' already exists. Appending data...")
    except:
        print(f"   Creating index '{INDEX_NAME}'...")
        client.create_index(uid=INDEX_NAME, options={'primaryKey': 'id'})
        index = client.index(INDEX_NAME)
        
        # Configure Settings (Run only once on creation)
        print("   Configuring attributes and ranking rules...")
        index.update_searchable_attributes([
            'company_name_cleaned_ascii', 'COMPANY_NAME', 'DOMAIN_PART', 
            'domain_parts_ngram', 'domain_phonetic', 'company_phonetic'
        ])
        
        index.update_ranking_rules([
            'words', 'typo', 'proximity', 'attribute', 'sort', 'exactness',
            'metadata_quality_score:desc', 
            'EMPLOYEE_COUNT:desc'
        ])
        
        # Facets for filtering
        index.update_filterable_attributes(['COUNTRY', 'INDUSTRY_CAT_STD', 'SOURCE'])
        
    return client.index(INDEX_NAME)

# --- 3. S3 & INGESTION PIPELINE ---

def ingest_from_s3():
    s3 = boto3.client('s3', region_name=AWS_REGION)
    index = init_meilisearch()
    
    # List all parquet files
    print(f"📂 Listing files in s3://{AWS_BUCKET}/{AWS_PREFIX}...")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=AWS_BUCKET, Prefix=AWS_PREFIX)
    
    file_keys = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.parquet'):
                    file_keys.append(obj['Key'])
    
    print(f"   Found {len(file_keys)} Parquet files.")
    
    total_processed = 0
    
    for idx, key in enumerate(file_keys):
        print(f"\n⬇️  [{idx+1}/{len(file_keys)}] Downloading {key}...")
        
        # Download file to memory buffer (Parquet is usually small enough per file)
        # If individual files are >1GB, we need a different strategy.
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        file_content = io.BytesIO(obj['Body'].read())
        
        # Read Parquet
        try:
            df = pd.read_parquet(file_content)
            print(f"   Read {len(df):,} rows from Parquet.")
        except Exception as e:
            print(f"❌ Error reading {key}: {e}")
            continue

        # Process and Upload in Batches
        # processing whole DF at once is faster for Vectorization, 
        # but we split for Upload.
        
        print("   Processing data (cleaning, phonetics)...")
        # Process the whole dataframe in memory
        documents = process_dataframe(df)
        
        # Send to Meili in chunks
        for i in range(0, len(documents), UPLOAD_BATCH_SIZE):
            batch = documents[i : i + UPLOAD_BATCH_SIZE]
            try:
                index.add_documents(batch)
                total_processed += len(batch)
                print(f"   Example: {batch[0]['company_name_cleaned_ascii']}") # Debug print
                print(f"   Sent batch {i//UPLOAD_BATCH_SIZE + 1} ({len(batch)} docs). Total: {total_processed:,}")
            except Exception as e:
                print(f"❌ Error uploading batch: {e}")
                # Optional: Retry logic here
                time.sleep(5)
                
    print(f"\n✅ Ingestion Complete! Total documents sent: {total_processed:,}")

if __name__ == "__main__":
    ingest_from_s3()