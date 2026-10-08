import uvicorn

if __name__ == "__main__":
    print("Starting the Intelligent Medical Image Analysis Backend...")
    # This runs the FastAPI app located in app/main.py
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
