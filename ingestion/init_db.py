import os
import psycopg
from dotenv import load_dotenv

def init_db():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not found in .env")
        return

    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    print("Connecting to Neon Postgres...")
    try:
        # Use psycopg to connect and execute the schema
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()
            print("Successfully initialized database schema!")
    except Exception as e:
        print(f"Database initialization failed: {e}")

if __name__ == "__main__":
    init_db()
