from fastapi import FastAPI, APIRouter

app = FastAPI(
    title="Heartbeat",
    description="Heartbeat endpoint",
    version="1.0"
)
router = APIRouter()

@router.get("/heartbeat")
def heartbeat():
    return {"status": "healthy"}

app.include_router(router)