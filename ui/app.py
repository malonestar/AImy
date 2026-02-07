from flask import Flask, render_template, Response, request, jsonify, redirect
import time
import os
import threading
import argparse
import subprocess
import queue
import requests
from services.vision.frame_broadcast import get_jpeg_frame
import services.vision.frame_broadcast as fb
#print("[UI sees frame_broadcast at]", fb.__file__)
#print("[UI sees frame_broadcast module id]", id(fb))
from .mic_level import mic_level_queue
from collections import deque
import config

LOG_BUFFER_SIZE = 200
log_buffer = deque(maxlen=LOG_BUFFER_SIZE)


state_queue = queue.Queue()
def push_state_change(payload):
    state = payload.get("state")
    if state:
        state_queue.put(state)

chat_queue = queue.Queue()
CHAT_BUFFER_SIZE = 50
chat_buffer = deque(maxlen=CHAT_BUFFER_SIZE)

def push_chat_message(role, text):
    msg = {"role": role, "text": text}
    chat_queue.put(msg)
    chat_buffer.append(msg)   

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "assistant.log")

def open_browser(dev_mode=False):
    time.sleep(1.5)
    chromium_cmd = [
        "chromium",
        "--app=http://127.0.0.1:5000",
        "--start-maximized",
        "--disable-infobars",
        "--disable-session-crashed-bubble",
        "--noerrdialogs",
    ]
    subprocess.Popen(chromium_cmd)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return "ok"

# ---------------- LOG STREAM ----------------
def stream_logs():
    while not os.path.exists(LOG_FILE):
        time.sleep(0.5)

    with open(LOG_FILE, "r") as f:
        f.seek(0, os.SEEK_END)
        last_keepalive = time.time()

        while True:
            line = f.readline()
            if line:
                clean = line.rstrip()
                log_buffer.append(clean)     
                yield f"data: {clean}\n\n"
                last_keepalive = time.time()
            else:
                if time.time() - last_keepalive > 10:
                    yield ": keepalive\n\n"
                    last_keepalive = time.time()
                time.sleep(0.1)

@app.route("/logs")
def logs():
    def generate():
        for line in log_buffer:
            yield f"data: {line}\n\n"
        yield from stream_logs()
    return Response(generate(), mimetype="text/event-stream")

# ---------------- STATE STREAM ----------------
@app.route("/state")
def state_stream():
    def gen():
        while True:
            try:
                state = state_queue.get(timeout=1)
                yield f"data: {state}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(gen(), mimetype="text/event-stream")


@app.route("/push_state", methods=["POST"])
def push_state():
    data = request.get_json(force=True, silent=True) or {}
    state = data.get("state")
    if state:
        state_queue.put(state)
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 400


# ---------------- MIC STREAM ----------------
@app.route("/mic")
def mic_stream():
    def generate():
        while True:
            try:
                level = mic_level_queue.get(timeout=1)
                yield f"data: {level:.3f}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(generate(), mimetype="text/event-stream")


# ---------------- CHAT STREAM (NEW) ----------------

@app.route("/chat/stream")
def chat_stream():
    def generate():
        # replay existing chat
        for msg in chat_buffer:
            yield f"data: {msg['role']}|{msg['text']}\n\n"

        # live stream
        while True:
            try:
                msg = chat_queue.get(timeout=1)
                yield f"data: {msg['role']}|{msg['text']}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"
    return Response(generate(), mimetype="text/event-stream")

# ---------------- CHAT SUBMIT (MOVED UP) ----------------
MAIN_API = "http://127.0.0.1:7000"  # main.py server

@app.route("/chat", methods=["POST"])
def chat_submit():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False}), 400

    try:
        requests.post(
            f"{MAIN_API}/chat",
            json={"text": text, "source": "text"},
            timeout=1.0,
        )
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/push_chat", methods=["POST"])
def push_chat():
    data = request.get_json(force=True)
    msg = {"role": data["role"], "text": data["text"]}
    chat_queue.put(msg)
    chat_buffer.append(msg)
    return {"ok": True}

# ----------------- HEADER BUTTONS ------------
@app.route("/wake", methods=["POST"])
def ui_wake():
    requests.post(f"{MAIN_API}/wake", timeout=0.2)
    return "", 204

@app.route("/shutdown", methods=["POST"])
def ui_shutdown():
    requests.post(f"{MAIN_API}/shutdown", timeout=0.2)
    return "", 204

# ---------------- VIDEO FEED ----------------
@app.route("/video_feed")
def video_feed():
    return redirect("http://127.0.0.1:7000/video_feed", code=302)

#----------------- VIDEO FEED ROI EDIT ---------------
@app.route("/roi/edit", methods=["POST"])
def roi_edit():
    requests.post(
        "http://127.0.0.1:7000/vision/roi/edit",
        timeout=1.0
    )
    return {"ok": True}


# ---------------- MAIN ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    threading.Thread(
        target=open_browser,
        kwargs={"dev_mode": args.dev},
        daemon=True
    ).start()

    app.run(host="127.0.0.1", port=5000, debug=False)
