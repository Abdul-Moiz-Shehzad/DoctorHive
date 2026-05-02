from src.database import engine
from sqlalchemy import text

def run_migration():
    with engine.connect() as conn:
        print("Checking if preferred_model column exists...")
        result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='preferred_model';"))
        if not result.fetchone():
            print("Adding preferred_model column to users table...")
            conn.execute(text("ALTER TABLE users ADD COLUMN preferred_model VARCHAR DEFAULT 'gemini';"))
            conn.commit()
            print("Column added successfully.")
        else:
            print("Column already exists.")

if __name__ == "__main__":
    run_migration()
