from dotenv import load_dotenv
import os
from openai import OpenAI
import google.generativeai as genai

# Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client_gpt = OpenAI(api_key=OPENAI_API_KEY) 
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.0-flash")
neurology_knowledge_base_location = "src/app/knowledge_base/neurology_doctorhive_gemini_db"
cardiology_knowledge_base_location = "src/app/knowledge_base/cardiology_doctorhive_gemini_db"
ophthalmology_knowledge_base_location = "src/app/knowledge_base/ophthalmology_doctorhive_gemini_db"

# Create Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Database
#DATABASE_URL = os.getenv("DATABASE_URL")