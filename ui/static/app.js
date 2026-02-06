const consoleEl = document.getElementById("console");
const face = document.getElementById("face");

const logSource = new EventSource("/logs");
const stateSource = new EventSource("/state");

const mic = document.getElementById("mic-indicator");
const bars = mic ? mic.querySelectorAll(".mic-bar") : [];
const micSource = new EventSource("/mic");

let micInterval = null;
let micLevel = 0;

const wakeBtn = document.getElementById("btn-wake");
const shutdownBtn = document.getElementById("btn-shutdown");

// ---------------- WAKE BUTTON DISABLE ------------
function setWakeEnabled(enabled) {
  if (!wakeBtn) return;
  wakeBtn.disabled = !enabled;
  wakeBtn.classList.toggle("disabled", !enabled);
}

// ---------------- LOG STREAM ----------------
logSource.onmessage = (event) => {
  const line = document.createElement("div");
  line.textContent = event.data;
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
};

// ---------------- FACE STATE ----------------
function setFaceState(state) {
  if (!face) return;
  const faceContainer = face.closest(".face");
  if (!faceContainer) return;

  faceContainer.dataset.state = state;
}


// ---------------- MIC ANIMATION ----------------
micSource.onmessage = (event) => {
  micLevel = Math.max(0, Math.min(1, parseFloat(event.data)));
};

let smoothedLevel = 0;

function animateMic(level = 0.4) {
  smoothedLevel = smoothedLevel * 0.7 + level * 0.3;

  bars.forEach((bar, i) => {
    const variance = (Math.random() - 0.5) * 0.3;
    const scale = Math.min(1.4, 0.3 + smoothedLevel + variance);
    bar.style.transform = `scaleY(${scale})`;
  });
}


function stopMic() {
  if (micInterval) {
    clearInterval(micInterval);
    micInterval = null;
  }
  bars.forEach(bar => {
    bar.style.transform = "scaleY(0.3)";
  });
}

// ---------------- STATE STREAM ----------------
stateSource.onmessage = (event) => {
  const state = event.data.trim();
  console.log("[STATE]", state);

  setFaceState(state);

  // ---- wake button gating ----
  setWakeEnabled(state === "LOOKING");

  if (state === "LISTENING") {
    if (!micInterval && bars.length > 0) {
      micInterval = setInterval(() => animateMic(), 120);
    }
  } else {
    stopMic();
  }
};


const chatInput = document.getElementById("chat-input");
const chatSend  = document.getElementById("chat-send");

function submitChat() {
  const text = chatInput.value.trim();
  if (!text) return;

  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });

  chatInput.value = "";
}

chatSend.addEventListener("click", submitChat);

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitChat();
});

// ---------------- CHAT STREAM ----------------
document.addEventListener("DOMContentLoaded", () => {
  const chatHistory = document.querySelector(".chat-history");
  if (!chatHistory) {
    console.warn("[CHAT] .chat-history not found");
    return;
  }

  const chatSource = new EventSource("/chat/stream");

  chatSource.onmessage = (event) => {
    const data = event.data;
    if (!data || !data.includes("|")) return;

    const [role, text] = data.split("|", 2);
    if (!text) return;

    const msg = document.createElement("div");
    msg.className = `chat-msg ${role}`;
    msg.textContent = text;

    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  };

  chatSource.onerror = (err) => {
    console.error("[CHAT SSE ERROR]", err);
  };
});

// ---------------- ROI EDIT BUTTON ----------------
const roiBtn = document.getElementById("roi-edit");
if (roiBtn) {
  roiBtn.addEventListener("click", () => {
    fetch("/roi/edit", { method: "POST" });
  });
}

// ----------------- HEADER BUTTONS -----------------
// "Hey AImy" to trigger wakeword detected and voice pipeline
document.getElementById("btn-wake").onclick = () => {
  fetch("/wake", { method: "POST" });
};

// "shutdown" to gracefully exit using existing processes
document.getElementById("btn-shutdown").onclick = () => {
  if (!confirm("Shut down AImy?")) return;
  fetch("/shutdown", { method: "POST" });
};


