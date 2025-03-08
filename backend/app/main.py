from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.cv_routes import router as cv_router
from app.api.query_routes import router as query_router

app = FastAPI(title="CV Analysis System API")

app.include_router(cv_router)
app.include_router(query_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
