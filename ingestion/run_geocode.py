import os
import io
import csv
import sys
import time
import requests
import psycopg
from dotenv import load_dotenv

CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
BATCH_SIZE = 1000  # Census API limit per request


def geocode_batch(rows, test_mode=False):
    """
    rows: list of (facility_id, address, city, state, zip)
    Returns a dict: { facility_id -> {"lat": float, "lon": float} or None }
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    for facility_id, address, city, state, zipc in rows:
        writer.writerow([facility_id, address, city, state, zipc or ""])
    payload_csv = buf.getvalue()

    resp = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                CENSUS_BATCH_URL,
                data={"benchmark": "Public_AR_Current", "vintage": "Current_Current"},
                files={"addressFile": ("addresses.csv", payload_csv.encode("utf-8"), "text/csv")},
                timeout=120,
            )
            if resp.status_code == 200:
                break
            print(f"Census API error (Status {resp.status_code}) on attempt {attempt+1}: {resp.text[:100]}...")
        except Exception as e:
            print(f"Census API request failed on attempt {attempt+1}: {e}")
            
        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))  # Backoff: 5s, 10s
    else:
        return {}  # Failed after all retries

    if test_mode:
        print(f"\n--- TEST MODE: Raw API Response Head ---")
        print("\n".join(resp.text.splitlines()[:5]))
        print("----------------------------------------\n")

    results = {}
    reader = csv.reader(io.StringIO(resp.text))
    for row in reader:
        if len(row) < 6:
            continue
        facility_id = row[0].strip()
        match_indicator = row[2].strip()  # "Match", "No_Match", "Tie"
        
        if match_indicator == "Match":
            coords = row[5].strip()  
            try:
                lon_str, lat_str = coords.split(",")
                results[facility_id] = {"lat": float(lat_str), "lon": float(lon_str)}
            except Exception:
                results[facility_id] = None
        else:
            results[facility_id] = None

    return results


def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not found in .env")
        return

    test_mode = "--test" in sys.argv
    if test_mode:
        print("Running in TEST MODE (processing only 10 records).")

    print("Connecting to Postgres database for geocoding...")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            limit_clause = "LIMIT 10" if test_mode else ""
            cur.execute(f"""
                SELECT facility_id::text, address, city, source_state, zip
                FROM facilities
                WHERE has_geo = false
                  AND address IS NOT NULL
                  AND city IS NOT NULL
                ORDER BY source_state, created_at DESC
                {limit_clause}
            """)
            rows = cur.fetchall()

    total = len(rows)
    print(f"Found {total} facilities needing geocoding.")
    if total == 0:
        print("Nothing to do.")
        return

    # Split into batches
    batches = [rows[i: i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"Processing in {len(batches)} batch(es) of up to {BATCH_SIZE}...")

    success_count = 0
    no_match_count = 0

    for batch_num, batch in enumerate(batches, 1):
        print(f"  Geocoding batch {batch_num}/{len(batches)} ({len(batch)} addresses)...")

        geo_results = geocode_batch(batch, test_mode=test_mode)

        if test_mode:
            print("\n--- TEST MODE: Parsed Results ---")
            for k, v in list(geo_results.items())[:5]:
                print(f"{k}: {v}")
            print("---------------------------------\n")

        # Open a fresh Postgres connection for each batch to prevent idle timeouts
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for facility_id, address, city, state, zipc in batch:
                    geo = geo_results.get(facility_id)
                    if geo:
                        cur.execute("""
                            UPDATE facilities
                            SET latitude = %s,
                                longitude = %s,
                                has_geo = true,
                                is_geocoded_fallback = true
                            WHERE facility_id = %s::uuid
                        """, (geo["lat"], geo["lon"], facility_id))
                        success_count += 1
                    else:
                        no_match_count += 1

                conn.commit()
                
        print(f"    Batch {batch_num} done. Running total — Matched: {success_count}, No match: {no_match_count}")

        # Be polite between batches
        if batch_num < len(batches) and not test_mode:
            time.sleep(2)

    print(f"\nGeocoding complete.")
    print(f"Successfully backfilled: {success_count} / {total}")
    print(f"No match found:          {no_match_count} / {total}")


if __name__ == "__main__":
    main()
