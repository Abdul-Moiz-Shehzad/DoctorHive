from src.database import engine
from sqlalchemy import text

def run_migration():
    with engine.connect() as conn:
        # Check chat_history
        print("Checking chat_history columns...")
        res = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='chat_history';"))
        columns = [r[0] for r in res.fetchall()]
        print(f"Current columns: {columns}")
        
        if 'chat_history' in columns: # wait, 'chat_history' is the table name, columns should be like 'id', 'user_id'
            pass

        if 'user_id' not in columns:
            print("Adding user_id to chat_history...")
            conn.execute(text("ALTER TABLE chat_history ADD COLUMN user_id INTEGER REFERENCES users(id);"))
        
        if 'case_id' not in columns:
            print("Adding case_id to chat_history...")
            conn.execute(text("ALTER TABLE chat_history ADD COLUMN case_id VARCHAR REFERENCES cases(case_id);"))

        if 'snapshot' not in columns:
            print("Adding snapshot to chat_history...")
            conn.execute(text("ALTER TABLE chat_history ADD COLUMN snapshot JSONB DEFAULT '{}';"))

        if 'updated_at' not in columns:
            print("Adding updated_at to chat_history...")
            conn.execute(text("ALTER TABLE chat_history ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"))

        conn.commit()
        print("Migration complete.")

if __name__ == "__main__":
    run_migration()
