run:
	pip install -r requirements.txt
	ngrok config add-authtoken <your-authtoken>
	python3 run_server.py
