# CNN API Project

This project trains a simple CNN model and builds a FastAPI app to make image predictions.

## How it works
- First, I trained a CNN model in `main.py` and saved it as `cnn.pth`.
- Then, I used FastAPI in `app/main.py` to load the model and create two endpoints:
  - `/` → check if the API is running
  - `/predict` → upload an image and get the predicted class

## How to run
1. Clone the repository  
```bash
git clone https://github.com/Jovnic/Convolutional-Neural-Networks.git
cd Convolutional-Neural-Networks

2. Set up the environment
uv sync

# if you wanna train the module yourself
# uv run python main.py

3. start the api
uv run uvicorn app.main:app --reload

4.check on the web
http://127.0.0.1:8000/docs

