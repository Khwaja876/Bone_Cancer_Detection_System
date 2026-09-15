import os
import sys
import webbrowser
import threading
import time
import uvicorn

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

def open_browser():
    time.sleep(2)
    webbrowser.open(f"http://{HOST}:{PORT}")

if __name__ == "__main__":
    print(f"🚀 Starting Bone Cancer Detection App at http://{HOST}:{PORT}")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=False)
