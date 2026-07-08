import os
import sys
import time
import psycopg
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
EMBED_BATCH_SIZE = 100  # texts per OpenAI API call
QDRANT_UPSERT_BATCH = 100  # points per Qdrant upsert call
COLLECTION_NAME = "facilities_v1"

# --- Payload fields: copied verbatim from Postgres into each Qdrant point ---
PAYLOAD_FIELDS = [
    # Identity & location
    "source_state", "data_source", "facility_type", "name",
    "city", "county", "zip", "latitude", "longitude",
    # Ratings
    "overall_rating", "health_inspection_rating", "staffing_rating", "quality_measure_rating",
    # Services
    "pt_service", "ot_service", "speech_service", "iv_service", "dme_service",
    "hospice_service", "social_work_service", "home_health_aide_service", "provides_nursing_care",
    # Staffing
    "rn_hours_per_resident_day", "lpn_hours_per_resident_day", "cna_hours_per_resident_day",
    "total_nursing_hours_per_resident_day", "rn_turnover_pct", "total_nursing_turnover_pct",
    "pt_hours_per_resident_day", "administrators_left_12mo",
    "staff_stability", "staffing_level_assessment",
    # Safety
    "total_fines_usd", "number_of_fines", "abuse_complaint", "special_focus_facility",
    "infection_control_citations", "health_deficiencies_count",
    "total_penalties", "weighted_health_inspection_score", "health_deficiency_severity_score",
    "medicare_payment_denials",
    # Outcomes
    "improved_walking_mobility_pct", "improved_bathing_ability_pct", "falls_major_injury_pct",
    "hospital_readmission_flag", "home_discharge_flag",
    "improved_breathing_pct", "improved_getting_out_of_bed_pct",
    "improved_taking_medications_pct", "started_care_on_time_pct",
    "medication_issues_fixed_on_time_pct", "info_shared_with_doctor_pct",
    "info_shared_with_family_pct", "functional_ability_discharge_score",
    "medicare_cost_vs_national_avg", "avoidable_hospitalizations_pct",
    # Ownership
    "ownership_type", "chain_affiliation",
    # Size & flags
    "bed_count", "ccrc_flag", "sprinkler_system_installed",
    "has_ratings", "has_staffing_data", "has_service_data", "has_geo",
]


def build_description(row: dict) -> str:
    """
    Generate a natural-language paragraph describing a facility.
    Only appends a sentence if the underlying data is non-null.
    Never writes 'Not Available' or 'None' into the text.
    """
    parts = []

    # --- Always included ---
    name = row.get("name") or "Unknown Facility"
    ftype = row.get("facility_type") or "Healthcare Facility"
    city = row.get("city") or "Unknown City"
    state = row.get("source_state") or ""
    parts.append(f"{name} is a {ftype} in {city}, {state}.")

    if row.get("address"):
        parts.append(f"Located at {row['address']}.")

    # --- Ownership & structure ---
    if row.get("chain_affiliation"):
        parts.append(f"Operated by {row['chain_affiliation']}.")
    if row.get("ownership_type"):
        parts.append(f"Ownership: {row['ownership_type']}.")
    if row.get("bed_count"):
        parts.append(f"{row['bed_count']} licensed beds.")
    if row.get("ccrc_flag"):
        parts.append("Continuing Care Retirement Community (CCRC).")

    # --- Ratings ---
    if row.get("has_ratings") and row.get("overall_rating"):
        parts.append(f"Overall Medicare rating: {row['overall_rating']} out of 5 stars.")
    if row.get("health_inspection_rating"):
        parts.append(f"Health inspection rating: {row['health_inspection_rating']} stars.")
    if row.get("staffing_rating"):
        parts.append(f"Staffing rating: {row['staffing_rating']} stars.")
    if row.get("quality_measure_rating"):
        parts.append(f"Quality measure rating: {row['quality_measure_rating']} stars.")

    # --- Services ---
    service_map = {
        "pt_service": "Physical Therapy",
        "ot_service": "Occupational Therapy",
        "speech_service": "Speech Therapy",
        "iv_service": "IV Therapy",
        "dme_service": "Durable Medical Equipment",
        "hospice_service": "Hospice services",
        "social_work_service": "Social Work services",
        "home_health_aide_service": "Home Health Aide services",
        "provides_nursing_care": "Skilled Nursing Care",
    }
    offered = [label for key, label in service_map.items() if row.get(key)]
    if offered:
        parts.append("Provides " + ", ".join(offered) + ".")

    # --- Staffing quality ---
    if row.get("has_staffing_data"):
        staffing_parts = []
        if row.get("rn_hours_per_resident_day") is not None:
            staffing_parts.append(f"{row['rn_hours_per_resident_day']} RN hours")
        if row.get("total_nursing_hours_per_resident_day") is not None:
            staffing_parts.append(f"{row['total_nursing_hours_per_resident_day']} total nursing hours")
        if staffing_parts:
            parts.append(" and ".join(staffing_parts) + " per resident day.")

        if row.get("rn_turnover_pct") is not None:
            parts.append(f"RN turnover rate: {row['rn_turnover_pct']}%.")
        if row.get("staff_stability"):
            parts.append(f"Staff stability: {row['staff_stability']}.")
        if row.get("staffing_level_assessment"):
            parts.append(f"Staffing assessment: {row['staffing_level_assessment']}.")
        if row.get("administrators_left_12mo") and row["administrators_left_12mo"] > 0:
            parts.append(f"{row['administrators_left_12mo']} administrators left in the past 12 months.")

    # --- Safety & compliance ---
    if row.get("total_fines_usd") and row["total_fines_usd"] > 0:
        parts.append(f"Total fines: ${row['total_fines_usd']:,.0f}.")
    if row.get("health_deficiencies_count") and row["health_deficiencies_count"] > 0:
        parts.append(f"{row['health_deficiencies_count']} health deficiencies in latest inspection.")
    if row.get("infection_control_citations") and row["infection_control_citations"] > 0:
        parts.append(f"{row['infection_control_citations']} infection control citations.")
    if row.get("abuse_complaint"):
        parts.append("Has an abuse complaint on record.")
    if row.get("special_focus_facility"):
        parts.append("Designated as a Special Focus Facility by CMS due to persistent quality concerns.")
    if row.get("penalty_summary"):
        parts.append(row["penalty_summary"])

    # --- Home Health outcomes ---
    if row.get("improved_walking_mobility_pct") is not None:
        parts.append(f"Walking mobility improvement rate: {row['improved_walking_mobility_pct']}%.")
    if row.get("improved_bathing_ability_pct") is not None:
        parts.append(f"Bathing ability improvement rate: {row['improved_bathing_ability_pct']}%.")
    if row.get("improved_breathing_pct") is not None:
        parts.append(f"Breathing improvement rate: {row['improved_breathing_pct']}%.")
    if row.get("improved_getting_out_of_bed_pct") is not None:
        parts.append(f"Getting out of bed improvement rate: {row['improved_getting_out_of_bed_pct']}%.")
    if row.get("falls_major_injury_pct") is not None:
        parts.append(f"Fall rate with major injury: {row['falls_major_injury_pct']}%.")
    if row.get("hospital_readmission_flag"):
        parts.append(f"Hospital readmission rate: {row['hospital_readmission_flag']}.")
    if row.get("home_discharge_flag"):
        parts.append(f"Home discharge success: {row['home_discharge_flag']}.")
    if row.get("medicare_cost_vs_national_avg") is not None:
        parts.append(f"Medicare cost compared to national average: {row['medicare_cost_vs_national_avg']}x.")

    return " ".join(parts)


def embed_texts(openai_client, texts):
    """Send a batch of texts to OpenAI and return their embedding vectors."""
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def fetch_all_facilities(db_url):
    """Pull every facility from Postgres as a list of dicts."""
    # Build the SELECT columns: facility_id + all payload fields + description-only fields
    extra_cols = ["address", "penalty_summary", "has_staffing_data"]
    all_cols = ["facility_id"] + PAYLOAD_FIELDS + [c for c in extra_cols if c not in PAYLOAD_FIELDS]
    col_str = ", ".join(all_cols)

    print("Fetching all facilities from Postgres...")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {col_str} FROM facilities ORDER BY source_state, name;")
            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()

    facilities = []
    for row in rows:
        d = {}
        for i, col in enumerate(columns):
            val = row[i]
            # Convert Decimal types to float for JSON serialization in Qdrant payload
            if hasattr(val, "as_integer_ratio"):
                val = float(val)
            d[col] = val
        facilities.append(d)

    print(f"Fetched {len(facilities)} facilities.")
    return facilities


def main():
    load_dotenv()

    db_url = os.getenv("DATABASE_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    for name, val in [("DATABASE_URL", db_url), ("OPENAI_API_KEY", openai_key),
                      ("QDRANT_URL", qdrant_url), ("QDRANT_API_KEY", qdrant_key)]:
        if not val:
            print(f"Error: {name} not found in .env")
            return

    test_mode = "--test" in sys.argv
    if test_mode:
        print("Running in TEST MODE (processing only 20 records).\n")

    openai_client = OpenAI(api_key=openai_key)
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)

    # Step 1: Fetch from Postgres
    facilities = fetch_all_facilities(db_url)
    if test_mode:
        facilities = facilities[:20]

    # Step 2: Build descriptions
    print("Building text descriptions...")
    descriptions = []
    for fac in facilities:
        desc = build_description(fac)
        descriptions.append(desc)

    if test_mode:
        print("\n--- TEST MODE: Sample Descriptions ---")
        for i in range(min(3, len(descriptions))):
            print(f"\n[{facilities[i]['name']}]")
            print(descriptions[i])
        print("--------------------------------------\n")

    # Step 3: Create/recreate Qdrant collection
    print(f"Creating Qdrant collection '{COLLECTION_NAME}'...")
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    # Step 4: Embed + upsert in batches
    total = len(facilities)
    upserted = 0

    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, total)
        batch_facilities = facilities[batch_start:batch_end]
        batch_descriptions = descriptions[batch_start:batch_end]

        # Embed
        vectors = embed_texts(openai_client, batch_descriptions)

        # Build Qdrant points
        points = []
        for i, fac in enumerate(batch_facilities):
            fac_id = str(fac["facility_id"])

            # Build payload: only include non-None values
            payload = {}
            for field in PAYLOAD_FIELDS:
                val = fac.get(field)
                if val is not None:
                    payload[field] = val

            points.append(PointStruct(
                id=fac_id,
                vector=vectors[i],
                payload=payload,
            ))

        # Upsert to Qdrant with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True,
                )
                break
            except Exception as e:
                print(f"  Qdrant upsert error on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

        upserted += len(points)
        print(f"  Embedded and upserted {upserted} / {total} points...")

    # Step 5: Verify
    qdrant_count = qdrant_client.count(collection_name=COLLECTION_NAME).count
    print(f"\nSync complete!")
    print(f"Qdrant point count: {qdrant_count}")
    print(f"Postgres row count: {total}")

    if qdrant_count == total:
        print("PASS: Counts match perfectly.")
    else:
        print(f"WARNING: Count mismatch! Qdrant has {qdrant_count}, expected {total}.")


if __name__ == "__main__":
    main()
