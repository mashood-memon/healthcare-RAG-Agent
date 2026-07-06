import os
import sys
import json
import yaml
import psycopg
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Mapping of states to their raw CSV files
STATE_FILES = {
    "NC": "cleaned_file - cleaned_file.csv",
    "CO": "merged_care_facilities_CO - merged_care_facilities_CO.csv",
    "AZ": "merged_care_facilities_AZ (2) - merged_care_facilities_AZ (2).csv",
    "CA": "merged_care_facilities_CA - merged_care_facilities_CA.csv"
}

NC_SOURCE_FILE_TO_DATA_SOURCE = {
    "health_care_north_carolina.csv": "directory",
    "Nursing_homes_NC.csv": "cms",
    "Home_Health_NC.csv": "cms",
    "hospice_NC.csv": "cms",
    "Inpatient_Rehabilitation_NC.csv": "cms",
}

def normalize_key(name, address, state):
    n = str(name).strip().upper() if pd.notnull(name) else ""
    a = str(address).strip().upper() if pd.notnull(address) else ""
    s = str(state).strip().upper() if pd.notnull(state) else ""
    return f"{n}|{a}|{s}"

def parse_value(val, dtype):
    if pd.isnull(val):
        return None
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in ['data not available', 'not available', 'null', 'nan']:
        return None

    if dtype == 'boolean':
        if val_str.lower() in ['yes', 'true', '1', 'y', 't', 'checked']:
            return True
        if val_str.lower() in ['no', 'false', '0', 'n', 'f']:
            return False
        return None
    elif dtype in ['numeric', 'integer', 'smallint', 'double precision']:
        # Remove commas, $ signs etc
        clean_num = val_str.replace(',', '').replace('$', '').replace('%', '')
        try:
            if dtype == 'integer' or dtype == 'smallint':
                return int(float(clean_num))
            return float(clean_num)
        except ValueError:
            return None
    elif dtype == 'date':
        try:
            return pd.to_datetime(val_str).date()
        except:
            return None
    return val_str

def apply_crosswalk(row, crosswalk_dict, state_code):
    record = {}
    row_dict = row.to_dict()
    
    for field, config in crosswalk_dict['fields'].items():
        if field in ['facility_id', 'has_ratings', 'has_staffing_data', 'has_service_data', 'has_geo', 'is_geocoded_fallback', 'raw_source', 'natural_key', 'created_at', 'updated_at']:
            continue
            
        sources = config.get('sources', {})
        state_source = sources.get(state_code)
        
        dtype = config.get('dtype', 'text')
        val = None
        
        if not state_source:
            val = None
        elif isinstance(state_source, list):
            for s in state_source:
                if s in row_dict and pd.notnull(row_dict[s]):
                    v = str(row_dict[s]).strip()
                    if v and v.lower() not in ['data not available', 'not available', 'null', 'nan']:
                        val = row_dict[s]
                        break
        else:
            if state_source in row_dict:
                val = row_dict[state_source]
                
        record[field] = parse_value(val, dtype)
        
        # CMS sometimes uses '9' (or other numbers) to represent 'Not Available' for star ratings
        if field.endswith('_rating') and record[field] not in [1, 2, 3, 4, 5, None]:
            record[field] = None
        
    # Bug 1 Fix: Explicit classification mapping instead of substring guessing
    if state_code == "NC":
        source_file_val = row_dict.get("SOURCE_FILE")
        record["data_source"] = NC_SOURCE_FILE_TO_DATA_SOURCE.get(source_file_val, "directory")
    else:
        ds = str(record.get("data_source")).lower() if record.get("data_source") else ""
        if ds.startswith("cms"):
            record["data_source"] = "cms"
        else:
            record["data_source"] = "directory"

    if not record.get("source_state"):
        record["source_state"] = state_code

    if not record.get("facility_type"):
        record["facility_type"] = "Unknown"
        
    if not record.get("name"):
        record["name"] = "Unknown Facility"
            
    # Derived flags
    record["has_ratings"] = record.get("overall_rating") is not None
    record["has_staffing_data"] = record.get("rn_hours_per_resident_day") is not None
    record["has_service_data"] = (record.get("data_source") == "cms")
    record["has_geo"] = (record.get("latitude") is not None and record.get("longitude") is not None)
    record["is_geocoded_fallback"] = False
    
    record["natural_key"] = normalize_key(record.get("name"), record.get("address"), state_code)
    
    # Store raw original row
    safe_dict = {}
    for k, v in row_dict.items():
        if pd.isnull(v):
            safe_dict[k] = None
        elif isinstance(v, (int, float, bool, str)):
            safe_dict[k] = v
        else:
            safe_dict[k] = str(v)
            
    record["raw_source"] = json.dumps(safe_dict)
    
    return record

def main():
    if len(sys.argv) < 2:
        print("Usage: python load_state.py <STATE_CODE>")
        sys.exit(1)
        
    state_code = sys.argv[1].upper()
    if state_code not in STATE_FILES:
        print(f"Error: Unknown state code {state_code}. Expected one of {list(STATE_FILES.keys())}")
        sys.exit(1)
        
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not found in .env")
        sys.exit(1)
        
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    csv_path = os.path.join(base_dir, 'data', 'raw', STATE_FILES[state_code])
    yaml_path = os.path.join(base_dir, 'data', 'crosswalk', 'crosswalk.yaml')
    
    print(f"Loading {state_code} data from {csv_path}...")
    
    with open(yaml_path, 'r') as f:
        crosswalk_dict = yaml.safe_load(f)
        
    df = pd.read_csv(csv_path, dtype=str)
    
    rows_read = len(df)
    rows_written = 0
    rejected = []
    
    print(f"Connecting to Postgres database...")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ingestion_runs (source_state, source_file, rows_read, rows_written, rows_rejected, rejection_reasons)
                VALUES (%s, %s, %s, 0, 0, '[]'::jsonb)
                RETURNING run_id
            """, (state_code, STATE_FILES[state_code], rows_read))
            run_id = cur.fetchone()[0]
            conn.commit()
            
            print(f"Started ingestion run {run_id} for {rows_read} rows.")
            
            cols = [k for k in crosswalk_dict['fields'].keys() if k != 'facility_id']
            cols = [c for c in cols if c not in ['has_ratings', 'has_staffing_data', 'has_service_data', 'has_geo', 'is_geocoded_fallback', 'raw_source', 'natural_key', 'created_at', 'updated_at']]
            cols.extend(['has_ratings', 'has_staffing_data', 'has_service_data', 'has_geo', 'is_geocoded_fallback', 'raw_source', 'natural_key'])
            
            placeholders = ", ".join([f"%({c})s" for c in cols])
            updates = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols if c != 'natural_key'])
            
            upsert_query = f"""
                INSERT INTO facilities ({", ".join(cols)})
                VALUES ({placeholders})
                ON CONFLICT (natural_key) DO UPDATE SET {updates}, updated_at = now();
            """
            
            for index, row in df.iterrows():
                try:
                    record = apply_crosswalk(row, crosswalk_dict, state_code)
                    
                    for c in cols:
                        if c not in record:
                            record[c] = None
                            
                    # Bug 2 Fix: Use conn.transaction() which maps to a SAVEPOINT in psycopg3
                    with conn.transaction():
                        cur.execute(upsert_query, record)
                    
                    rows_written += 1
                    
                    if rows_written % 1000 == 0:
                        print(f"Processed {rows_written} / {rows_read} rows...")
                        
                except Exception as e:
                    rejected.append({"index": index, "reason": str(e)})
                    
            conn.commit()
            
            cur.execute("""
                UPDATE ingestion_runs 
                SET rows_written = %s, rows_rejected = %s, rejection_reasons = %s, finished_at = now()
                WHERE run_id = %s
            """, (rows_written, len(rejected), json.dumps(rejected), run_id))
            conn.commit()
            
            print(f"Ingestion complete for {state_code}.")
            print(f"Written: {rows_written} | Rejected: {len(rejected)}")
            if len(rejected) > 0:
                print("First 3 rejection reasons for debugging:")
                for r in rejected[:3]:
                    print(r)

if __name__ == "__main__":
    main()
