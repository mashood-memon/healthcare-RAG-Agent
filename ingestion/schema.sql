CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;  -- for radius/near-me queries

DROP TABLE IF EXISTS facilities;
CREATE TABLE facilities (
    facility_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_state TEXT NOT NULL,
    source_file TEXT,
    data_source TEXT NOT NULL CHECK (data_source IN ('directory','cms')),
    facility_type TEXT NOT NULL,
    name TEXT NOT NULL,
    legal_business_name TEXT,
    ccn TEXT,
    license_number TEXT,
    address TEXT, 
    city TEXT, 
    county TEXT, 
    zip TEXT, 
    phone TEXT,
    latitude DOUBLE PRECISION, 
    longitude DOUBLE PRECISION,
    ownership_type TEXT,
    certification_date DATE,
    cms_region TEXT,
    bed_count INTEGER,
    overall_rating SMALLINT CHECK (overall_rating BETWEEN 1 AND 5),
    health_inspection_rating SMALLINT,
    staffing_rating SMALLINT,
    quality_measure_rating SMALLINT,
    pt_service BOOLEAN, 
    ot_service BOOLEAN, 
    speech_service BOOLEAN,
    iv_service BOOLEAN, 
    dme_service BOOLEAN, 
    hospice_service BOOLEAN,
    social_work_service BOOLEAN, 
    home_health_aide_service BOOLEAN,
    rn_hours_per_resident_day NUMERIC,
    lpn_hours_per_resident_day NUMERIC,
    cna_hours_per_resident_day NUMERIC,
    total_nursing_hours_per_resident_day NUMERIC,
    rn_turnover_pct NUMERIC,
    total_nursing_turnover_pct NUMERIC,
    total_fines_usd NUMERIC,
    number_of_fines INTEGER,
    abuse_complaint BOOLEAN,
    special_focus_facility BOOLEAN,
    infection_control_citations INTEGER,
    health_deficiencies_count INTEGER,
    improved_walking_mobility_pct NUMERIC,
    improved_bathing_ability_pct NUMERIC,
    falls_major_injury_pct NUMERIC,
    hospital_readmission_flag TEXT,
    home_discharge_flag TEXT,
    chain_affiliation TEXT,
    owner_name TEXT, 
    mgmt_company_name TEXT, 
    administrator_name TEXT,
    
    -- Additional canonical fields from crosswalk
    improved_breathing_pct NUMERIC,
    improved_getting_out_of_bed_pct NUMERIC,
    improved_taking_medications_pct NUMERIC,
    started_care_on_time_pct NUMERIC,
    medication_issues_fixed_on_time_pct NUMERIC,
    info_shared_with_doctor_pct NUMERIC,
    info_shared_with_family_pct NUMERIC,
    functional_ability_discharge_score NUMERIC,
    medicare_cost_vs_national_avg NUMERIC,
    avoidable_hospitalizations_pct NUMERIC,
    provides_nursing_care BOOLEAN,
    pt_hours_per_resident_day NUMERIC,
    average_daily_residents NUMERIC,
    administrators_left_12mo INTEGER,
    staff_stability TEXT,
    staffing_level_assessment TEXT,
    ccrc_flag BOOLEAN,
    sprinkler_system_installed BOOLEAN,
    medicare_payment_denials INTEGER,
    total_penalties NUMERIC,
    penalty_summary TEXT,
    weighted_health_inspection_score NUMERIC,
    health_deficiency_severity_score NUMERIC,

    -- derived flags — computed at ingestion, NEVER inferred at query time
    has_ratings BOOLEAN NOT NULL,
    has_staffing_data BOOLEAN NOT NULL,
    has_service_data BOOLEAN NOT NULL,
    has_geo BOOLEAN NOT NULL,
    is_geocoded_fallback BOOLEAN NOT NULL DEFAULT FALSE,
    raw_source JSONB,
    natural_key TEXT NOT NULL,  -- normalized name+address+state, for upsert dedup
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (natural_key)
);

CREATE INDEX idx_fac_state_type ON facilities(source_state, facility_type);
CREATE INDEX idx_fac_rating ON facilities(overall_rating) WHERE overall_rating IS NOT NULL;
CREATE INDEX idx_fac_geo ON facilities USING gist (ll_to_earth(latitude, longitude)) WHERE has_geo;
CREATE INDEX idx_fac_flags ON facilities(has_ratings, has_service_data, has_geo);

DROP TABLE IF EXISTS ingestion_runs;
CREATE TABLE ingestion_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_state TEXT,
    source_file TEXT,
    rows_read INTEGER,
    rows_written INTEGER,
    rows_rejected INTEGER,
    rejection_reasons JSONB,
    started_at TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ
);

DROP TABLE IF EXISTS query_logs;
CREATE TABLE query_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT,
    classification JSONB,
    sql_executed TEXT,
    rows_returned INTEGER,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);
