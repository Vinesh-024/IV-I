import logging
import sqlite3
import os
import json
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from app.config import settings

logger = logging.getLogger("app.database")

# Global clients and DB references
mongo_client = None
db = None
use_fallback_db = False
FALLBACK_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fallback_medical.db")

async def init_db():
    global mongo_client, db, use_fallback_db
    
    logger.info("Initializing database connection...")
    try:
        # Try connecting to MongoDB with a short timeout (2 seconds)
        mongo_client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000)
        # Trigger server selection to force verification of connection
        await mongo_client.server_info()
        db = mongo_client[settings.DB_NAME]
        logger.info(f"Successfully connected to MongoDB at {settings.MONGO_URI}, DB: {settings.DB_NAME}")
        use_fallback_db = False
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB: {e}. Falling back to SQLite local database.")
        use_fallback_db = True
        init_fallback_sqlite()

def init_fallback_sqlite():
    conn = sqlite3.connect(FALLBACK_DB_PATH)
    cursor = conn.cursor()
    # patient_reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_reports (
            id TEXT PRIMARY KEY,
            patient_name TEXT,
            patient_age INTEGER,
            patient_sex TEXT,
            symptoms TEXT,
            medical_history TEXT,
            image_filename TEXT,
            modality TEXT,
            body_part_scan_type TEXT,
            disease TEXT,
            confidence REAL,
            severity TEXT,
            specialist TEXT,
            report_text TEXT,
            rag_suggestions TEXT,
            mask_filename TEXT,
            gradcam_filename TEXT,
            created_at TEXT
        )
    """)
    # users table (Phase 2 auth)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            full_name TEXT,
            email TEXT UNIQUE,
            hashed_password TEXT,
            clinic_name TEXT,
            role TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"SQLite fallback database initialized at {FALLBACK_DB_PATH}")

async def save_report(report_data: dict) -> str:
    report_id = report_data.get("id")
    if not report_id:
        raise ValueError("Report must contain an id")
    
    if not use_fallback_db:
        # Save to MongoDB
        collection = db["patient_reports"]
        report_write = dict(report_data)
        if "_id" in report_write:
            del report_write["_id"]
        await collection.update_one({"id": report_id}, {"$set": report_write}, upsert=True)
        return report_id
    else:
        # Save to SQLite fallback
        conn = sqlite3.connect(FALLBACK_DB_PATH)
        cursor = conn.cursor()
        
        # Flatten dictionary elements for SQL storage
        cursor.execute("""
            INSERT OR REPLACE INTO patient_reports (
                id, patient_name, patient_age, patient_sex, symptoms, medical_history,
                image_filename, modality, body_part_scan_type, disease, confidence,
                severity, specialist, report_text, rag_suggestions,
                mask_filename, gradcam_filename, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id,
            report_data.get("patient_name"),
            report_data.get("patient_age"),
            report_data.get("patient_sex"),
            report_data.get("symptoms"),
            report_data.get("medical_history"),
            report_data.get("image_filename"),
            report_data.get("modality"),
            report_data.get("body_part_scan_type"),
            report_data.get("disease"),
            report_data.get("confidence"),
            report_data.get("severity"),
            report_data.get("specialist"),
            report_data.get("report_text"),
            json.dumps(report_data.get("rag_suggestions", [])),
            report_data.get("mask_filename", ""),
            report_data.get("gradcam_filename", ""),
            report_data.get("created_at", datetime.now().isoformat())
        ))
        conn.commit()
        conn.close()
        return report_id

async def get_all_reports() -> list:
    if not use_fallback_db:
        collection = db["patient_reports"]
        reports = []
        async for doc in collection.find().sort("created_at", -1).limit(100):
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            reports.append(doc)
        return reports
    else:
        conn = sqlite3.connect(FALLBACK_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patient_reports ORDER BY created_at DESC")
        rows = cursor.fetchall()
        reports = []
        for row in rows:
            report = dict(row)
            # Unpack JSON suggestions
            try:
                report["rag_suggestions"] = json.loads(report["rag_suggestions"])
            except:
                report["rag_suggestions"] = []
            reports.append(report)
        conn.close()
        return reports

async def get_report_by_id(report_id: str) -> dict:
    if not use_fallback_db:
        collection = db["patient_reports"]
        doc = await collection.find_one({"id": report_id})
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc
    else:
        conn = sqlite3.connect(FALLBACK_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patient_reports WHERE id = ?", (report_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            report = dict(row)
            try:
                report["rag_suggestions"] = json.loads(report["rag_suggestions"])
            except:
                report["rag_suggestions"] = []
            return report
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — User / Auth CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def create_user(user_data: dict) -> str:
    user_id = user_data.get("id")
    if not use_fallback_db:
        collection = db["users"]
        await collection.insert_one(dict(user_data))
    else:
        conn = sqlite3.connect(FALLBACK_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO users
                (id, full_name, email, hashed_password, clinic_name, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_data.get("full_name"),
            user_data.get("email"),
            user_data.get("hashed_password"),
            user_data.get("clinic_name", ""),
            user_data.get("role", "Doctor"),
            user_data.get("created_at", datetime.now().isoformat())
        ))
        conn.commit()
        conn.close()
    return user_id


async def get_user_by_email(email: str) -> dict:
    email = email.lower().strip()
    if not use_fallback_db:
        collection = db["users"]
        doc = await collection.find_one({"email": email})
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc
    else:
        conn = sqlite3.connect(FALLBACK_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None


async def get_user_by_id(user_id: str) -> dict:
    if not use_fallback_db:
        collection = db["users"]
        doc = await collection.find_one({"id": user_id})
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc
    else:
        conn = sqlite3.connect(FALLBACK_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
