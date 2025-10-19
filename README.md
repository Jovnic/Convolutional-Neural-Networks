# CNN API Project

This project trains a simple CNN model and builds a FastAPI app to make predictions from images.

## How it works
- First, I trained a CNN model in `main.py` and saved it as `cnn.pth`.
- Then, I used FastAPI in `app/main.py` to load the model and create two endpoints:
  - `/` → to check if the API is running
  - `/predict` → to upload an image and get the predicted class

## How to run
1. Train the model  
```bash
uv run python main.py

2. Start 
uv run uvicorn app.main:app --reload

3. check on the web
http://127.0.0.1:8000/docs
