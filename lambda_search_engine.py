import asyncio
import json
import logging
import time
import re
from typing import List, Dict, Any, Optional
from meilisearch_python_sdk import AsyncClient

# Try to import meilisearch-python-sdk for async support
# pip install meilisearch-python-sdk
# try:
#     from meilisearch_python_sdk import AsyncClient
# except ImportError:
#     # Fallback for local testing without async client (not recommended for prod)
#     import meilisearch as AsyncClient 
#     print("WARNING: Using sync client. Install 'meilisearch-python-sdk' for async speed.")

# --- CONFIGURATION ---
MEILI_URL = "http://localhost:7700" # Update for AWS (e.g., EC2 IP)
MEILI_KEY = "testMasterKey123"
INDEX_NAME = "companies"

# Configure Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- HELPER FUNCTIONS ---

def clean_text(text: str) -> str:
    """Standardize input text for matching."""
    if not text: return ""
    try:
        # Lowercase, strip, remove non-ascii
        text = text.lower().strip()
        text = text.encode('ascii', errors='ignore').decode('ascii')
        return text
    except:
        return ""

def get_phonetic(text: str) -> str:
    """
    Generate phonetic code (Simulated Metaphone).
    MUST match the logic used in 'data_preprocessing.py'.
    """
    if not text: return ""
    text = clean_text(text).upper()
    if not text: return ""
    
    first = text[0]
    remainder = text[1:]
    remainder = re.sub(r'[AEIOUY]', '', remainder)
    remainder = re.sub(r'[BFPV]', '1', remainder)
    remainder = re.sub(r'[CGJKQSXZ]', '2', remainder)
    remainder = re.sub(r'[DT]', '3', remainder)
    remainder = re.sub(r'[L]', '4', remainder)
    remainder = re.sub(r'[MN]', '5', remainder)
    remainder = re.sub(r'[R]', '6', remainder)
    
    # Remove adjacent duplicates
    last = ""
    reduced = ""
    for char in remainder:
        if char != last:
            reduced += char
        last = char
    return first + reduced

def get_ngrams(text: str, min_len=3) -> List[str]:
    """Generate n-grams for partial matching."""
    text = clean_text(text)
    if not text: return []
    # In search, we usually just query the text itself against the ngram field,
    # but if you want to search specific fragments, you can generate them here.
    # For now, we will just search the full cleaning string against the array.
    return [text] 

# --- TIERED SEARCH LOGIC ---

class CompanySearchEngine:
    def __init__(self, url: str, key: str, index_name: str):
        self.url = url
        self.key = key
        self.index_name = index_name

    async def _search_tier(self, index, query: str, tier_name: str, 
                         search_params: Dict[str, Any]) -> List[Dict]:
        """Generic async search wrapper for a specific tier."""
        try:
            # We explicitly ask for specific attributes to reduce payload size
            search_params['attributes_to_retrieve'] = [
                'company_name_cleaned_ascii', 'DOMAIN_NAME', 'EMPLOYEE_COUNT', 
                'metadata_quality_score', 'source_rank'
            ]
            
            # Execute Search
            results = await index.search(query, **search_params)
            
            # Tag results with the Tier that found them
            hits = results.hits 
            for hit in hits:
                hit['_match_tier'] = tier_name
                hit['_match_score'] = 0 # Will calculate later
            return hits
        except Exception as e:
            logger.error(f"Error in {tier_name} for '{query}': {str(e)}")
            return []

    async def search_company(self, client, raw_name: str) -> Dict[str, Any]:
        """
        Orchestrates the multi-tier parallel search for a SINGLE company name.
        """
        index = client.index(self.index_name)
        clean_name = clean_text(raw_name)
        phonetic_code = get_phonetic(clean_name)
        
        # --- DEFINE PARALLEL QUERIES ---
        
        tasks = []

        # Tier 1: Exact Match (High Confidence)
        # Search specifically in the cleaned name field
        tasks.append(self._search_tier(index, clean_name, "Tier 1 (Exact)", {
            'attributes_to_search_on': ['company_name_cleaned_ascii', 'COMPANY_NAME'],
            'limit': 1,
            'matching_strategy': 'all' # Must match all words
        }))

        # Tier 2: Domain Part Exact (High Confidence)
        # Check if the input string matches a domain part exactly
        tasks.append(self._search_tier(index, clean_name, "Tier 2 (Domain Exact)", {
            'attributes_to_search_on': ['DOMAIN_PART'],
            'limit': 1
        }))

        # Tier 3: Typo Tolerant (Medium Confidence)
        # Standard Meilisearch behavior (allows typos)
        tasks.append(self._search_tier(index, clean_name, "Tier 3 (Typo)", {
            'attributes_to_search_on': ['company_name_cleaned_ascii'],
            'limit': 3
        }))

        # Tier 4: Phonetic Match (Medium Confidence)
        # Search the pre-calculated phonetic codes
        if phonetic_code:
            tasks.append(self._search_tier(index, phonetic_code, "Tier 4 (Phonetic)", {
                'attributes_to_search_on': ['company_phonetic', 'domain_phonetic'],
                'limit': 3
            }))

        # Tier 5: Partial/N-gram Match (Low/Medium Confidence)
        # Search against the n-gram array we built
        tasks.append(self._search_tier(index, clean_name, "Tier 5 (N-gram)", {
            'attributes_to_search_on': ['domain_parts_ngram'],
            'limit': 5
        }))
        
        # Tier 7: Alternative Names (High Confidence)
        # Search in the alt names array
        tasks.append(self._search_tier(index, clean_name, "Tier 7 (Alt Names)", {
            'attributes_to_search_on': ['alternative_names'],
            'limit': 1
        }))

        # --- EXECUTE ALL TIERS IN PARALLEL ---
        
        # asyncio.gather runs all search requests concurrently
        results_list = await asyncio.gather(*tasks)
        
        # --- AGGREGATE & RANK RESULTS ---
        
        all_candidates = {}
        
        for tier_hits in results_list:
            for hit in tier_hits:
                domain = hit.get('DOMAIN_NAME')
                if not domain: continue
                
                # If we've seen this domain, keep the higher tier priority? 
                # Or just accumulate evidence.
                # Here, we keep the first occurrence (since tiers are somewhat prioritized by logic)
                if domain not in all_candidates:
                    all_candidates[domain] = hit
                    
                    # Score Calculation Logic (Replicating "DENSE_RANK")
                    # Start with Tier Base Score
                    score = 0
                    tier = hit['_match_tier']
                    if "Tier 1" in tier: score = 95
                    elif "Tier 2" in tier: score = 90
                    elif "Tier 7" in tier: score = 88
                    elif "Tier 3" in tier: score = 80
                    elif "Tier 4" in tier: score = 75
                    elif "Tier 5" in tier: score = 70
                    
                    # Boost by Metadata Quality (0-40 pts)
                    quality = hit.get('metadata_quality_score', 0)
                    score += (quality / 2) # Weighted boost
                    
                    # Boost by Size (Logarithmic boost)
                    emps = hit.get('EMPLOYEE_COUNT', 0) or 0
                    if emps > 10000: score += 10
                    elif emps > 1000: score += 5
                    
                    hit['_match_score'] = min(score, 100) # Cap at 100

        # Sort candidates by calculated score
        sorted_candidates = sorted(
            all_candidates.values(), 
            key=lambda x: x['_match_score'], 
            reverse=True
        )

        # Return best match or None
        if not sorted_candidates:
            return {
                "input_name": raw_name,
                "match_found": False,
                "confidence": 0,
                "details": "No match in any tier"
            }

        best_match = sorted_candidates[0]
        
        return {
            "input_name": raw_name,
            "match_found": True,
            "domain": best_match['DOMAIN_NAME'],
            "company_name": best_match.get('company_name_cleaned_ascii'),
            "tier": best_match['_match_tier'],
            "confidence": best_match['_match_score'],
            "candidates_found": len(sorted_candidates)
        }

    async def process_batch(self, company_names: List[str]) -> List[Dict]:
        """
        Process a list of 1000 names efficiently.
        """
        async with AsyncClient(self.url, self.key) as client:
            # Create a coroutine for each company name
            tasks = [self.search_company(client, name) for name in company_names]
            
            # Execute all 1000 searches concurrently
            # (Meilisearch handles high concurrency well, but we can chunk if needed)
            results = await asyncio.gather(*tasks)
            return results

# --- MAIN LAMBDA HANDLER MOCK ---

def lambda_handler(event, context):
    """
    AWS Lambda Entry Point
    Expects event to be: {"companies": ["Microsoft", "Apple", ...]}
    """
    start_time = time.time()
    
    company_names = event.get('companies', [])
    if not company_names:
        return {"statusCode": 400, "body": "No companies provided"}

    print(f"🚀 Processing {len(company_names)} companies...")
    
    # Initialize Engine
    engine = CompanySearchEngine(MEILI_URL, MEILI_KEY, INDEX_NAME)
    
    # Run Async Loop
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(engine.process_batch(company_names))
    
    duration = time.time() - start_time
    print(f"✅ Finished in {duration:.2f} seconds.")
    
    # Calculate Stats
    matched = sum(1 for r in results if r['match_found'])
    print(f"📊 Stats: {matched}/{len(company_names)} matched ({matched/len(company_names)*100:.1f}%)")
    
    return {
        "statusCode": 200,
        "body": json.dumps(results),
        "metrics": {
            "duration": duration,
            "processed": len(company_names),
            "matches": matched
        }
    }

# --- LOCAL TESTING ---
if __name__ == "__main__":
    # Test Payload
    test_event = {
        "companies": [
            "Microsoft",          # Tier 1 (Exact)
            "Microsft Corp",      # Tier 3 (Typo)
            "Maikrosoft",         # Tier 4 (Phonetic)
            "Micro",              # Tier 5 (N-gram) + Rank boost (Employees)
            "Big Blue",           # Tier 7 (Alt Name - if data exists)
            "Heal Within",        # From your sample
            "Heaney General"      # From your sample
        ]
    }
    
    response = lambda_handler(test_event, None)
    
    # Print readable results
    print("\n--- RESULTS ---")
    data = json.loads(response['body'])
    for res in data:
        status = "✅" if res['match_found'] else "❌"
        conf = f"{res['confidence']:.0f}%" if res['match_found'] else "0%"
        print(f"{status} {res['input_name']:<20} -> {res.get('domain', 'N/A'):<25} [{res.get('tier', '-')}] (Conf: {conf})")
