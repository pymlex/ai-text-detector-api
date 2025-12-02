run:
	pip install -q -r requirements.txt
	ngrok config add-authtoken <your-ngrok-authtoken>
	python3 run_server.py
