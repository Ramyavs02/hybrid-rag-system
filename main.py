from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # temporary for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
