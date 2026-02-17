# tasker_notify.py
import os
import json
import subprocess
import requests
import time

# --- Config you already use ---
PHONE_TAILNET_IP = os.environ.get("PHONE_TAILNET_IP", "100.66.151.92")
PHONE_HTTP_PORT = os.environ.get("PHONE_HTTP_PORT", "1821")
TASKER_ENDPOINT = os.environ.get("TASKER_ENDPOINT", "/general")  # whatever your Tasker HTTP endpoint is

# --- Minimal tailscale helpers ---
SUDO = os.environ.get("SUDO_BIN", "sudo")
TS_BIN = os.environ.get("TS_BIN", "tailscale")

def _run(cmd, check=True, timeout=15):
    return subprocess.run(cmd, check=check, text=True, capture_output=True, timeout=timeout)

def tailscale_is_up() -> bool:
    try:
        p = _run([TS_BIN, "status"], check=False)
        out = (p.stdout or "").lower()
        # crude but effective: "logged out" means down; if we see a self line or "logged in", call it up
        return ("logged out" not in out) and ("logged in" in out or "active;" in out or "100." in out)
    except Exception:
        return False

def tailscale_up():
    # exactly like your manual command
    _run([SUDO, TS_BIN, "up"])

def tailscale_down():
    # exactly like your manual command
    _run([SUDO, TS_BIN, "down"], check=False)

# --- Your existing sender (unchanged API) ---
def send_to_tasker(title: str, text: str) -> bool:
    url = f"http://{PHONE_TAILNET_IP}:{PHONE_HTTP_PORT}{TASKER_ENDPOINT}"
    payload = {"title": title, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[tasker_notify] send failed: {e}")
        return False

# --- One-call helper that ensures tailscale for the send, then restores state ---
def send_with_tailscale(title: str, text: str) -> bool:
    was_up = tailscale_is_up()
    brought_up = False
    try:
        if not was_up:
            print("[tasker_notify] tailscale up...")
            tailscale_up()
            brought_up = True
        time.sleep(1)
        # Try up to 3 times if sending fails
        for attempt in range(1, 4):
            ok = send_to_tasker(title, text)
            if ok:
                print(f"[tasker_notify] ✅ Success on attempt {attempt}")
                return True
            else:
                print(f"[tasker_notify] ⚠️ Attempt {attempt} failed, retrying...")
                time.sleep(2)  # small delay between retries

        print("[tasker_notify] ❌ All 3 attempts failed.")
        return False
    finally:
        if brought_up:
            print("[tasker_notify] tailscale down...")
            time.sleep(1)
            tailscale_down()

if __name__ == "__main__":
    # quick manual test:
    #   python3 tasker_notify.py "Ping" "Hello from Pi"
    import sys
    title = sys.argv[1] if len(sys.argv) > 1 else "Notify script"
    body  = sys.argv[2] if len(sys.argv) > 2 else "No title or body sent with script"
    print("sent:", send_with_tailscale(title, body))
