"""
Baymax Healthcare Companion - Master Application Launcher.
Starts the FastAPI backend server and launches the web interface.
"""

import sys
import time
import webbrowser
import uvicorn


def main():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print("=" * 65)
    print("      BAYMAX - TRUE HEALTHCARE COMPANION SYSTEM")
    print("=" * 65)
    print(f"[*] Starting Baymax Medical Cockpit on: {url}")
    print("[*] Non-Contact Perception Engine: rPPG (CHROM/POS) & Acoustics")
    print("[*] Clinical Intelligence: ESI v4 Deterministic Triage Safeguards")
    print("[*] Affective Interface: Voice Modulation & Empathic Avatar")
    print("=" * 65)

    # Launch browser after brief startup delay
    def open_browser():
        time.sleep(1.5)
        print(f"[*] Opening browser interface at {url}...")
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Uvicorn ASGI server
    try:
        uvicorn.run("backend.app:app", host=host, port=port, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\n[*] Baymax: 'I am satisfied with my care.' Shutting down safely.")
        sys.exit(0)


if __name__ == "__main__":
    main()
