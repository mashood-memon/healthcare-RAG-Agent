import os
import sys
import time
import requests
import psycopg
from dotenv import load_dotenv

def geocode_address(address, city, state, zip_code):
    time.sleep(1.5) # respect Nominatim limits
    
    # Bug 1 Fix: Filter out None values before joining to avoid 'None' literal in query
    parts = [p for p in [address, city, state, zip_code] if p]
    query = ", ".join(parts)
    
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "HealthcareFacilityDataIngestion/1.0 (valid_contact_antigravity@example.com)"}
    params = {"q": query, "format": "json", "limit": 1}
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"Nominatim returned non-200 (Status {resp.status_code}): {resp.text[:100]}")
            return "ERROR"
            
        data = resp.json()
        if data and len(data) > 0:
            return {
                "lat": float(data[0]['lat']),
                "lon": float(data[0]['lon'])
            }
    except Exception as e:
        print(f"Geocoding failed for {query}: {e}")
        return "ERROR"
        
    return None

def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not found in .env")
        sys.exit(1)
        
    print("Connecting to Postgres database for geocoding...")
    
    address_cache = {}
    consecutive_failures = 0
    
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT facility_id, address, city, source_state, zip
                FROM facilities 
                WHERE has_geo = false 
                  AND address IS NOT NULL 
                  AND city IS NOT NULL
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
            total = len(rows)
            print(f"Found {total} facilities needing geocoding.")
            
            success_count = 0
            for i, row in enumerate(rows):
                facility_id, address, city, state, zipc = row
                key = f"{address}|{city}|{state}|{zipc}".upper()
                
                # Bug 2 Fix: In-memory cache for duplicate addresses
                if key in address_cache:
                    geo = address_cache[key]
                else:
                    geo = geocode_address(address, city, state, zipc)
                    
                    # Operational Gap Fix: Hard stop on consecutive API errors (403/429)
                    if geo == "ERROR":
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            print("5 consecutive geocoding API errors — possible rate limit/block. Stopping early.")
                            break
                        geo = None # Treat this row as failed but try the next one
                    else:
                        consecutive_failures = 0
                        address_cache[key] = geo # Cache the valid response (or None if simply not found)
                
                if geo:
                    cur.execute("""
                        UPDATE facilities 
                        SET latitude=%s, longitude=%s, has_geo=true, is_geocoded_fallback=true 
                        WHERE facility_id=%s
                    """, (geo["lat"], geo["lon"], facility_id))
                    conn.commit()
                    success_count += 1
                
                if (i + 1) % 50 == 0:
                    hits = sum(1 for v in address_cache.values() if v is not None)
                    print(f"Processed {i + 1} / {total} addresses (Success: {success_count}, Unique Hits Cached: {hits})")
                    
            print(f"Geocoding run complete. Successfully backfilled {success_count} / {total} coordinates.")

if __name__ == "__main__":
    main()
