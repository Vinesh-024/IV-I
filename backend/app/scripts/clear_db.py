import os
import sqlite3
import pymongo

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_PATH = os.path.join(os.path.dirname(BASE_DIR), "fallback_medical.db")
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "medical_analysis"

def clear_mongo():
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        db = client[DB_NAME]
        db["patient_reports"].drop()
        print("[OK] MongoDB collection 'patient_reports' dropped successfully.")
    except Exception as e:
        print(f"[INFO] MongoDB connection failed or offline: {e}. Skipping Mongo cleanup.")

def clear_sqlite():
    if os.path.exists(SQLITE_PATH):
        try:
            conn = sqlite3.connect(SQLITE_PATH)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS patient_reports")
            conn.commit()
            conn.close()
            # Recreate table empty
            from app.database import init_db
            import asyncio
            asyncio.run(init_db())
            print("[OK] SQLite database table 'patient_reports' cleared successfully.")
        except Exception as e:
            print(f"[ERROR] SQLite cleanup failed: {e}")
    else:
        print("[INFO] No SQLite fallback database file found. Skipping SQLite cleanup.")

def main():
    print("=== Clearing Patient History Database ===")
    clear_mongo()
    clear_sqlite()
    print("Database reset complete! The application is now clean.")

if __name__ == "__main__":
    main()
