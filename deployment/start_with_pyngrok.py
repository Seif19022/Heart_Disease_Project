import sys, subprocess, time
from pathlib import Path
from pyngrok import ngrok
from pyngrok.conf import PyngrokConfig

# Streamlit entry point
APP_PATH  = Path("Heart_Disease_Project/ui/app.py").resolve()

# Put BOTH the ngrok binary and the ngrok config under your user profile
NGROK_BIN = Path(r"C:\Users\sseno\ngrokbin\ngrok.exe")  
CFG_PATH  = Path(r"C:\Users\sseno\ngrokcfg\ngrok.yml")

# Your token here
NGROK_TOKEN = "32MzUMaST5RzRWwaNAeyQrQMgtM_463mXFbwi7NTNrgRvdGVc"

# Ensure folders exist
NGROK_BIN.parent.mkdir(parents=True, exist_ok=True)
CFG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Write config once if missing
if not CFG_PATH.exists():
    CFG_PATH.write_text(f'version: "3"\nauthtoken: {NGROK_TOKEN}\n', encoding="utf-8")

# Tell pyngrok exactly where to install/use ngrok.exe + config
cfg = PyngrokConfig(
    ngrok_path=str(NGROK_BIN),  
    config_path=str(CFG_PATH),
    ngrok_version="v3"
)

# Start Streamlit with the current venv's Python
app = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", str(APP_PATH),
    "--server.address=127.0.0.1", "--server.port=8501"
])

# Open tunnel (first run: pyngrok will download ngrok.exe to NGROK_BIN)
public_url = ngrok.connect(8501, "http", pyngrok_config=cfg)
print("\nYour app is live at:", public_url, "\n(Press Ctrl+C to stop)\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping ...")
    ngrok.kill()
    app.terminate()
