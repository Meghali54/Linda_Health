from flask import Flask, jsonify, request
from itsdangerous import URLSafeTimedSerializer
import os
from urllib.parse import urlencode

app = Flask(__name__)

TOKEN_SECRET = os.getenv("TOKEN_SECRET", "dev-secret")
STREAMLIT_URL = os.getenv("STREAMLIT_URL", "https://linda-health.onrender.com")

serializer = URLSafeTimedSerializer(TOKEN_SECRET)

@app.route("/generate")
def generate():
    expires = int(request.args.get("expires", 3600))
    payload = {"user": "anonymous"}  # extend with user info if needed
    token = serializer.dumps(payload)
    link = f"{STREAMLIT_URL}?{urlencode({'token': token})}"
    return jsonify({"link": link, "expires_in": expires})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
