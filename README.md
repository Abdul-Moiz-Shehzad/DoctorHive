# DoctorHive: Advanced Multi-Agent Clinical Consultation Platform

## Project Overview
DoctorHive is a sophisticated medical consultation system designed to provide preliminary clinical assessments through a multi-agent artificial intelligence framework. The platform leverages large language models (LLMs) to simulate a professional clinical environment, guiding patients from initial symptom reporting to specialist consultation and diagnostic reasoning.

The system is engineered to prioritize clinical accuracy, patient data persistence, and transparency in AI decision-making (eXplainable AI - XAI).

## System Description
At the core of DoctorHive is a collaborative multi-agent architecture. The system features three primary specialist agents—dedicated to Cardiology, Neurology, and Ophthalmology—working in tandem with a central General Practitioner (GP) agent. These specialists are designed to collaborate within the orchestrator framework, sharing clinical insights and transitioning the consultation based on the patient's specific needs. To provide the most accurate and high-quality responses, the agents engage in dynamic, interactive Q&A sessions with the user, proactively seeking clarifying information to refine their assessment and ensure a comprehensive clinical evaluation.

## Current Implementation Status
The project has reached a stable production-ready state with the following core modules fully implemented:
*   **Authentication & Profile Management**: Secure user registration, login, and comprehensive patient profiling.
*   **GP Preliminary Assessment**: An initial screening agent that analyzes patient symptoms and medical history.
*   **Specialist Agent Network**: Dedicated agents for Cardiology, Neurology, Ophthalmology, and other medical disciplines.
*   **Orchestrator Logic**: A central control unit managing state transitions between clinical phases and agent referrals.
*   **XAI Reasoning Logs**: Real-time logging of the AI's internal reasoning process for clinical transparency.
*   **Session Persistence**: Complete state preservation for ongoing and historical consultations.

## System Architecture

### Backend
*   **Framework**: FastAPI (Python 3.10+)
*   **Database**: PostgreSQL with SQLAlchemy ORM
*   **Intelligence**: Integration with Google Gemini and OpenAI models
*   **Security**: Token-based authentication and CORS-compliant middleware

### Frontend
*   **Library**: React 19
*   **Styling**: Premium Vanilla CSS with Glassmorphic design principles
*   **Icons**: Lucide React
*   **State Management**: React Router for navigation and localized state for consultation flows

## Installation and Environment Setup

### Prerequisites
*   Python 3.10 or higher
*   Node.js (LTS version recommended)
*   PostgreSQL 14 or higher

### 1. Backend Configuration
Navigate to the root directory and create a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/macOS
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Frontend Configuration
Navigate to the frontend directory and install dependencies:
```bash
cd frontend
npm install
```

### 3. Environment Variables

#### Backend Configuration (`.env`)
Create a `.env` file in the root directory. You must define the following variables for the system to function. 

**Important Note on API Keys**: The system is designed to utilize multiple Gemini API keys to handle rate limiting across different agents. If you only possess a single API key, you must still define all four variables and assign the same key to each.

```env
# AI Service Keys
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_primary_gemini_key
GOOGLE_API_KEY_second=your_primary_gemini_key
GOOGLE_API_KEY_third=your_primary_gemini_key
GOOGLE_API_KEY_fourth=your_primary_gemini_key

# Database Configuration
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432

# Security & External Resources
BACKEND_FRONTEND_TOKEN=your_secure_random_token
RAG_REGISTRY_URL=https://gist.githubusercontent.com/.../rag_registry.json
```

#### Frontend Configuration (`frontend/.env`)
Create a `.env` file in the `frontend` directory to configure the React application's connection to the backend.

```env
# API Connection
REACT_APP_API_BASE=http://localhost:8000

# Security
# This token must match the BACKEND_FRONTEND_TOKEN defined in the backend .env
REACT_APP_BACKEND_TOKEN=your_secure_random_token
```


### 4. Database Setup
Ensure PostgreSQL is running and create a database matching the `DB_NAME` defined in your `.env` file. The application will handle table initialization and schema adjustments on the first startup.

## Running the Application

### Starting the Backend
From the root directory, execute the following command:
```bash
uvicorn src.app.routers.main:app --reload
```
The API documentation will be available at `http://localhost:8000/docs`.

### Starting the Frontend
From the `frontend` directory, execute:
```bash
npm start
```
The application will be accessible at `http://localhost:3000`.

## Operational Manual

### Consultation Workflow
1.  **Onboarding**: Users register and complete their medical profile (Allergies, Current Medications, Past History).
2.  **GP Assessment**: The user describes current symptoms. The GP Agent performs a preliminary analysis and determines if a specialist is required.
3.  **Specialist Referral**: If referred, the Orchestrator transitions the session to the relevant specialist (e.g., Cardiologist).
4.  **Clinical Conclusion**: The specialist provides a final assessment, recommended tests, and potential diagnoses.
5.  **Reasoning Review**: Throughout the process, the 'Reasoning' panel displays the internal logic used by the agents.

### Troubleshooting
*   **Database Connection Errors**: Verify that the PostgreSQL service is active and the credentials in `.env` are accurate.
*   **API Rate Limits**: If using a single Gemini key for all variables, be mindful of the concurrent request limits for the free tier.
*   **Token Mismatch**: Ensure the `BACKEND_FRONTEND_TOKEN` in the backend `.env` matches the configuration expected by the frontend.

## License
This project is licensed under the Apache License 2.0. Refer to the LICENSE file for full details.
