
# AI Text Detector API

<img width="1762" height="333" alt="image" src="https://github.com/user-attachments/assets/44d835d4-f350-4a3d-ba15-db96c813565b" />

A minimal API for the Desklib text-detection model from [Hugging Face](https://huggingface.co/desklib/ai-text-detector-v1.01). The service exposes a `/predict` endpoint, accepts `text` or `title + text` pairs, and returns the model’s probability and boolean label based on a configurable threshold.

The project keeps functionality intentionally simple: a **FastAPI** application for routing and validation, a compact `Detector` wrapper around the Hugging Face model, and optional **ngrok** integration for external access (useful in Colab). The internal model component handles tokenisation, batching, GPU/CPU device selection and probability computation.

## Project structure

- **app/main.py** – FastAPI application, CORS, endpoints `/health`, `/info`, `/predict`, threshold logic, batch handling.  
- **app/model.py** – model wrapper: tokenizer loading, PyTorch forward pass, probability computation, batching.  
- **run_server.py** – ngrok tunnel + uvicorn server.  
- **client_test.py** – minimal usage example for making prediction requests.  
 - **requirements.txt** – runtime dependencies.
 - **Makefile** - modules installation and ngrok runner. Set your `ngrok_authtoken` here.

## Request structure
  
JSON body for `POST /predict`:
```json
{
  "items": [
    {
      "id": 1,
      "title": "optional title",
      "text": "text to analyse"
    }
  ],
  "mode": "normal | strict | light",
  "threshold": 0.5,
  "batch_size": 16
}
````

You may provide either:

* `"items": [ ... ]` – list of objects
* `"item": { ... }` – single object

Only one of `mode` or `threshold` should be used.

Response:

```json
{
  "predictions": [
    {
      "id": 1,
      "probability": 0.1343,
      "label": false
    }
  ]
}
```

## Running in Google Colab

```bash
!git clone https://github.com/pymlex/ai-text-detector-api.git
%cd ai-text-detector-api
!make run
```

After execution, the notebook prints the status of trainformer installation and the server running, along with some additional information. Most importantly, it provides a **public URL** from ngrok that you can use for the /predict endpoint.

## Client usage example

Copy the URL from Colab and paste it into `client_test.py` on your local computer. Then, run the client:

```bash
py -3.12 client_test.py
```
