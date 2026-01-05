import pandas as pd
import numpy as np
import re
import json
import time

# --- CONFIGURATION ---
INPUT_FILE = "Sample_dataset.csv"      # Your raw data file
OUTPUT_FILE = "companies_processed.json" # Output for Meilisearch
BATCH_SIZE = 100000                  # Process in chunks to save memory

# --- 1. CLEANING & TEXT PROCESSING FUNCTIONS ---

def clean_text_basic(text):
    """
    Standardizes text: lowercase, ASCII only, removes special chars.
    Example: "Heal Within®" -> "heal within"
    """
    if not isinstance(text, str): 
        return ""
    # Remove non-ASCII characters (e.g., trademarks)
    try:
        text = text.encode('ascii', errors='ignore').decode('ascii')
    except:
        pass
    return text.lower().strip()

def get_ngrams(text, min_len=3, max_len=15):
    """
    Generates n-grams for partial matching (Tier 5).
    Input: "micro" -> ["mic", "micr", "micro"]
    """
    if not text or not isinstance(text, str): 
        return []
    
    text = clean_text_basic(text)
    if len(text) < min_len: 
        return [text]
    
    # Generate prefixes
    return [text[:i] for i in range(min_len, min(len(text) + 1, max_len + 1))]

def simple_phonetic(text):
    """
    Generates phonetic code for sound-alike matching (Tier 4).
    Simulates a simplified Soundex/Metaphone.
    """
    if not text or not isinstance(text, str): 
        return ""
    
    # 1. Standardize
    text = clean_text_basic(text).upper()
    if not text: 
        return ""
    
    # 2. Keep first letter
    first = text[0]
    remainder = text[1:]
    
    # 3. Phonetic Transformations
    # Remove vowels
    remainder = re.sub(r'[AEIOUY]', '', remainder)
    # Map consonants to sound groups
    remainder = re.sub(r'[BFPV]', '1', remainder)
    remainder = re.sub(r'[CGJKQSXZ]', '2', remainder)
    remainder = re.sub(r'[DT]', '3', remainder)
    remainder = re.sub(r'[L]', '4', remainder)
    remainder = re.sub(r'[MN]', '5', remainder)
    remainder = re.sub(r'[R]', '6', remainder)
    
    # 4. Squeeze adjacent duplicates (e.g., "11" -> "1")
    last_char = ""
    reduced = ""
    for char in remainder:
        if char != last_char:
            reduced += char
        last_char = char
        
    return first + reduced

def calc_quality_score(row):
    """
    Calculates a 'Metadata Quality Score' for Ranking (Tier 6).
    Replicates the logic of preferring rich, verified data.
    """
    score = 0
    
    # 1. Source Priority (PDL > BOMBORA > HGDATA)
    src = str(row.get('SOURCE', '')).upper()
    if 'PDL' in src: score += 20
    elif 'BOMBORA' in src: score += 15
    elif 'HGDATA' in src: score += 10
    
    # 2. Metadata Richness
    if pd.notna(row.get('EMPLOYEE_COUNT')) and row.get('EMPLOYEE_COUNT') > 0: 
        score += 10
    if pd.notna(row.get('INDUSTRY_CAT_STD')): 
        score += 5
    if pd.notna(row.get('COUNTRY')): 
        score += 2
    if pd.notna(row.get('SIZE_DESC_STD')): 
        score += 3
        
    # 3. Recency / Verification
    if pd.notna(row.get('LAST_SEEN_DATE')) or pd.notna(row.get('DATE_LAST_VERIFIED')):
        score += 5
        
    return score

def get_source_rank(source):
    """Simple integer rank for sorting (Lower is better)"""
    if not isinstance(source, str): return 99
    src = source.upper()
    if 'PDL' in src: return 1
    if 'BOMBORA' in src: return 2
    if 'HGDATA' in src: return 3
    return 4

# --- 2. MAIN PROCESSING PIPELINE ---

def process_chunk(chunk):
    """Processes a single chunk of the dataframe"""
    
    # 1. Basic Cleaning & ID
    # Use existing ID if present, otherwise we assume index handling outside
    # Ensure NaN strings are handled
    chunk = chunk.replace({np.nan: None})

    # 2. Text Normalization for Search Columns
    chunk['company_name_cleaned_ascii'] = chunk['COMPANY_NAME_CLEANED'].apply(clean_text_basic)
    
    # 3. Tier 5: N-grams (Partial Match)
    # Use DOMAIN_PART if available, else derive from DOMAIN_NAME
    if 'DOMAIN_PART' not in chunk.columns:
         chunk['DOMAIN_PART'] = chunk['DOMAIN_NAME'].astype(str).apply(lambda x: x.split('.')[0] if '.' in x else x)
    
    chunk['domain_parts_ngram'] = chunk['DOMAIN_PART'].apply(get_ngrams)
    
    # 4. Tier 4: Phonetics (Sound-alike)
    # Generate for both domain part and company name for max recall
    chunk['domain_phonetic'] = chunk['DOMAIN_PART'].apply(simple_phonetic)
    chunk['company_phonetic'] = chunk['COMPANY_NAME_CLEANED'].apply(simple_phonetic)
    
    # 5. Tier 6: Ranking Logic
    chunk['metadata_quality_score'] = chunk.apply(calc_quality_score, axis=1)
    chunk['source_rank'] = chunk['SOURCE'].apply(get_source_rank)
    
    # 6. Tier 7: Alternative Names (Placeholders)
    # In production, you would merge your alt_names DataFrame here.
    # Creating empty lists so the field exists in Meilisearch
    chunk['alternative_names'] = [[] for _ in range(len(chunk))]
    chunk['alternative_phonetic'] = [[] for _ in range(len(chunk))]
    
    # 7. Final JSON Clean
    # Convert dataframe to list of dicts, cleaning any non-serializable values
    records = chunk.to_dict(orient='records')
    cleaned_records = []
    
    for r in records:
        clean_r = {}
        for k, v in r.items():
            # Filter out None/NaN values to save space, 
            # Meilisearch treats missing fields as null anyway
            if v is not None and v != "" and not (isinstance(v, float) and np.isnan(v)):
                clean_r[k] = v
        cleaned_records.append(clean_r)
        
    return cleaned_records

def main():
    print(f"🚀 Starting Preprocessing for {INPUT_FILE}...")
    start_time = time.time()
    
    # Load and Process in Chunks to handle large file size (memory efficient)
    processed_count = 0
    
    # Initialize output file (overwrite)
    with open(OUTPUT_FILE, 'w') as f:
        f.write('[') # Start JSON array
    
    first_chunk = True
    
    # Read CSV in chunks
    for chunk in pd.read_csv(INPUT_FILE, chunksize=BATCH_SIZE, low_memory=False):
        
        # Add ID if missing (using global counter)
        if 'id' not in chunk.columns:
            chunk.insert(0, 'id', range(processed_count + 1, processed_count + 1 + len(chunk)))
            
        processed_data = process_chunk(chunk)
        
        # Write to file incrementally
        with open(OUTPUT_FILE, 'a') as f:
            for i, record in enumerate(processed_data):
                if not first_chunk or i > 0:
                    f.write(',') # Separator between objects
                json.dump(record, f)
            
        first_chunk = False
        processed_count += len(chunk)
        print(f"   Processed {processed_count:,} records...")
        break

    # Close JSON array
    with open(OUTPUT_FILE, 'a') as f:
        f.write(']')
        
    elapsed = time.time() - start_time
    print(f"✅ DONE! Processed {processed_count:,} records in {elapsed:.1f} seconds.")
    print(f"📁 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
