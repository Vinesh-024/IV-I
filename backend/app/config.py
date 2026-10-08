import os
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

class Settings:
    # MongoDB settings
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "medical_analysis")
    
    # Model configuration
    MODEL_DIR: str = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    DEVICE: str = os.getenv("DEVICE", "cuda") # will be auto-calibrated on load
    
    # LLM Settings (for Report and RAG)
    API_PROVIDER: str = os.getenv("API_PROVIDER", "gemini") # 'gemini', 'openai', or 'local'
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LOCAL_LLM_URL: str = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate") # Ollama URL
    
    # Port / Host Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()

# Ensure model directory exists
os.makedirs(settings.MODEL_DIR, exist_ok=True)
