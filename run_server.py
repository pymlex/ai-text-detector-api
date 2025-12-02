import requests
import uvicorn
from threading import Thread
from app.main import app


ip = requests.get("https://api.ipify.org").text
print("API_URL=http://" + ip + ":8000/predict")

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

Thread(target=run).start()
