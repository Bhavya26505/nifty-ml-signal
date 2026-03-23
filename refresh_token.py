"""
NiftyEdge — Upstox Token Refresher
Run this every morning before 9 AM on your laptop.
It opens a browser, you login to Upstox once,
token is saved automatically to GitHub Secrets.
Takes about 30 seconds total.
"""

import requests
import webbrowser
import json
import os
import subprocess
from flask import Flask, request
import threading
import time

# ══════════════════════════════════════════════
# FILL THESE IN — your Upstox app credentials
# ══════════════════════════════════════════════
API_KEY      = "46c61475-c41b-4acd-a3e9-89a11b6f7ae9"        # from upstox developer app
API_SECRET   = "jhhd6tyaqm"    # from upstox developer app
REDIRECT_URI = "http://127.0.0.1:5000/callback"

# ══════════════════════════════════════════════
# FILL THIS IN — your GitHub details
# ══════════════════════════════════════════════
GITHUB_TOKEN = "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" # see instructions below
GITHUB_OWNER = "Bhavya26505"
GITHUB_REPO  = "nifty-ml-signal"
SECRET_NAME  = "UPSTOX_ACCESS_TOKEN"

# ═══════════════════════════════════════════════
# DO NOT EDIT BELOW THIS LINE
# ═══════════════════════════════════════════════

app = Flask(__name__)
captured_code = {"value": None}
shutdown_flag = {"done": False}

@app.route("/callback")
def callback():
    code = request.args.get("code")
    if code:
        captured_code["value"] = code
        shutdown_flag["done"] = True
        return """
        <html><body style='font-family:sans-serif;text-align:center;padding:60px;background:#0d1117;color:#3fb950'>
        <h2>Login successful!</h2>
        <p style='color:#7d8590'>Token captured. You can close this tab.</p>
        <p style='color:#7d8590'>Go back to your terminal to see the result.</p>
        </body></html>
        """
    return "Error: no code received", 400

def run_flask():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

def get_access_token(auth_code):
    url = "https://api.upstox.com/v2/login/authorization/token"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "code":          auth_code,
        "client_id":     API_KEY,
        "client_secret": API_SECRET,
        "redirect_uri":  REDIRECT_URI,
        "grant_type":    "authorization_code",
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"Token exchange failed: {response.status_code} — {response.text}")
        return None

def save_token_to_github(token):
    """
    Save the access token to GitHub Secrets using GitHub API.
    Requires a Personal Access Token with repo + secrets scope.
    """
    import base64
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
    from cryptography.hazmat.primitives import serialization
    import nacl.public
    import nacl.encoding

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # Get repo public key for encryption
    key_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/secrets/public-key"
    key_resp = requests.get(key_url, headers=headers)
    if key_resp.status_code != 200:
        print(f"Could not get GitHub public key: {key_resp.status_code}")
        return False

    key_data = key_resp.json()
    public_key_b64 = key_data["key"]
    key_id = key_data["key_id"]

    # Encrypt the token using libsodium sealed box
    public_key_bytes = base64.b64decode(public_key_b64)
    public_key = nacl.public.PublicKey(public_key_bytes)
    sealed_box = nacl.public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(token.encode("utf-8"))
    encrypted_b64 = base64.b64encode(encrypted).decode("utf-8")

    # Save to GitHub Secrets
    secret_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/secrets/{SECRET_NAME}"
    secret_resp = requests.put(secret_url, headers=headers, json={
        "encrypted_value": encrypted_b64,
        "key_id": key_id
    })

    if secret_resp.status_code in [201, 204]:
        print(f"  Token saved to GitHub Secrets as '{SECRET_NAME}'")
        return True
    else:
        print(f"  GitHub save failed: {secret_resp.status_code} — {secret_resp.text}")
        return False

def save_token_locally(token):
    """Fallback — save token to local file if GitHub save fails"""
    with open("upstox_token.json", "w") as f:
        json.dump({
            "access_token": token,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print("  Token saved locally to upstox_token.json")

def main():
    print()
    print("=" * 55)
    print("  NiftyEdge — Upstox Token Refresher")
    print("=" * 55)
    print()

    # Check credentials filled in
    if "YOUR_" in API_KEY or "YOUR_" in API_SECRET:
        print("ERROR: Please fill in API_KEY and API_SECRET at the top of this file.")
        print("Get them from: https://account.upstox.com/developer/apps")
        return

    # Start local Flask server in background
    print("[1/4] Starting local callback server on port 5000...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(1.5)

    # Build auth URL and open browser
    auth_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code"
        f"&client_id={API_KEY}"
        f"&redirect_uri={REDIRECT_URI}"
    )

    print("[2/4] Opening Upstox login in your browser...")
    print(f"      URL: {auth_url}")
    print()
    print("      >> Login with your Upstox credentials")
    print("      >> After login you will be redirected back automatically")
    print()
    webbrowser.open(auth_url)

    # Wait for callback
    print("[3/4] Waiting for login callback...")
    timeout = 120
    elapsed = 0
    while not shutdown_flag["done"] and elapsed < timeout:
        time.sleep(0.5)
        elapsed += 0.5

    if not captured_code["value"]:
        print("ERROR: Login timed out after 2 minutes. Please try again.")
        return

    print(f"      Authorization code received!")

    # Exchange code for token
    print("[4/4] Exchanging code for access token...")
    access_token = get_access_token(captured_code["value"])

    if not access_token:
        print("ERROR: Could not get access token.")
        return

    print(f"      Access token obtained: {access_token[:25]}...")

    # Save to GitHub Secrets
    print()
    print("Saving token to GitHub Secrets...")
    if "YOUR_" not in GITHUB_TOKEN:
        try:
            saved = save_token_to_github(access_token)
            if not saved:
                save_token_locally(access_token)
        except ImportError:
            print("  PyNaCl not installed — saving locally instead")
            print("  Run: pip install pynacl")
            save_token_locally(access_token)
        except Exception as e:
            print(f"  GitHub save error: {e}")
            save_token_locally(access_token)
    else:
        print("  GitHub token not set — saving locally only")
        save_token_locally(access_token)

    print()
    print("=" * 55)
    print("  Done! Token is ready for today.")
    print("  GitHub Actions will use it automatically.")
    print("=" * 55)
    print()

if __name__ == "__main__":
    main()
