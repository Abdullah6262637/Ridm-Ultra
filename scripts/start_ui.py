import subprocess
import time

print("Starting FastAPI Uvicorn Server...")
api_proc = subprocess.Popen(["uv", "run", "uvicorn", "--factory", "ridm_ultra.api:create_app", "--port", "8000"])

print("Waiting 3 seconds for API to initialize...")
time.sleep(3)

print("Starting Streamlit App...")
ui_proc = subprocess.Popen(["uv", "run", "streamlit", "run", "ridm_ultra/ui/app.py", "--server.port", "8501"])

try:
    api_proc.wait()
except KeyboardInterrupt:
    api_proc.terminate()
    ui_proc.terminate()
