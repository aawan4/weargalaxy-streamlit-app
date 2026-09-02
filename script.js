const API_BASE = "";

const $ = (id) => document.getElementById(id);

const state = {
  mode: "webcam",
  stream: null,
  capturedBlob: null,
  uploadedFile: null,
  busy: false
};

const tabs = document.querySelectorAll(".mode-tab");
const modeContents = document.querySelectorAll(".mode-content");

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 3500);
}

function setLoading(loading) {
  state.busy = loading;
  $("loading").classList.toggle("hidden", !loading);
  $("resultCard").classList.toggle("hidden", loading);
}

function showResult(title, text) {
  $("resultTitle").textContent = title;
  $("resultText").textContent = text;
  setLoading(false);
}

function resetResult() {
  showResult(
    "Ready when you are",
    "Use your webcam, upload an image, or choose your face shape to get a personalized WeAR AI recommendation."
  );
}

function activateMode(mode) {
  state.mode = mode;

  tabs.forEach(tab => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });

  modeContents.forEach(content => {
    content.classList.toggle("active", content.id === `${mode}Mode`);
  });

  const isChat = mode === "chat";
  $("advisorSection").classList.toggle("hidden", isChat);
  $("chatSection").classList.toggle("hidden", !isChat);

  if (mode !== "webcam") {
    stopCamera();
  }
}

tabs.forEach(tab => {
  tab.addEventListener("click", () => activateMode(tab.dataset.mode));
});

$("mobileMenu").addEventListener("click", () => {
  document.querySelector(".nav-links").classList.toggle("open");
});

async function parseApiResponse(response) {
  let data = null;

  try {
    data = await response.json();
  } catch {
    throw new Error(`Server returned HTTP ${response.status}.`);
  }

  if (!response.ok) {
    const detail = data?.detail || data?.message || "Request failed.";
    throw new Error(detail);
  }

  return data;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Your browser does not support webcam access.");
    return;
  }

  try {
    stopCamera();

    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    const video = $("camera");
    video.srcObject = state.stream;
    video.classList.add("ready");
    $("cameraPlaceholder").classList.add("hidden");
    $("captureBtn").disabled = false;

    showToast("Camera started.");
  } catch (error) {
    showToast("Camera access was blocked. Please allow camera permission.");
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }

  const video = $("camera");
  video.srcObject = null;
  video.classList.remove("ready");
  $("cameraPlaceholder").classList.remove("hidden");
  $("captureBtn").disabled = true;
}

$("startCameraBtn").addEventListener("click", startCamera);

$("captureBtn").addEventListener("click", async () => {
  const video = $("camera");

  if (!video.videoWidth || !video.videoHeight) {
    showToast("Camera is not ready yet.");
    return;
  }

  const canvas = $("cameraCanvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    if (!blob) {
      showToast("Could not capture the photo.");
      return;
    }

    state.capturedBlob = blob;
    $("capturePreview").src = URL.createObjectURL(blob);
    $("capturePreviewWrap").classList.remove("hidden");
  }, "image/jpeg", 0.9);
});

$("analyzeCaptureBtn").addEventListener("click", async () => {
  if (!state.capturedBlob) {
    showToast("Capture a photo first.");
    return;
  }

  const form = new FormData();
  form.append("file", state.capturedBlob, "webcam.jpg");

  await analyzeMultipart(form);
});

$("imageInput").addEventListener("change", event => {
  const file = event.target.files?.[0];

  if (!file) return;

  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showToast("Please choose a JPG, PNG, or WEBP image.");
    event.target.value = "";
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showToast("Image is larger than 10 MB.");
    event.target.value = "";
    return;
  }

  state.uploadedFile = file;

  $("uploadPreview").src = URL.createObjectURL(file);
  $("uploadPreviewWrap").classList.remove("hidden");
});

$("analyzeUploadBtn").addEventListener("click", async () => {
  if (!state.uploadedFile) {
    showToast("Choose an image first.");
    return;
  }

  const form = new FormData();
  form.append("file", state.uploadedFile, state.uploadedFile.name);

  await analyzeMultipart(form);
});

async function analyzeMultipart(form) {
  if (state.busy) return;

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: form
    });

    const data = await parseApiResponse(response);

    showResult(
      "Analysis Complete",
      data.analysis || "WeAR AI returned no analysis."
    );
  } catch (error) {
    showResult("Analysis Failed", error.message);
    showToast(error.message);
  }
}

$("manualBtn").addEventListener("click", async () => {
  const shape = $("shapeSelect").value;

  if (!shape) {
    showToast("Please select your face shape.");
    return;
  }

  if (state.busy) return;

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/api/suggestion`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ shape })
    });

    const data = await parseApiResponse(response);

    showResult(
      `Your Face Shape Is: ${data.face_shape || shape}`,
      data.recommendation || "No recommendation returned."
    );
  } catch (error) {
    showResult("Recommendation Failed", error.message);
    showToast(error.message);
  }
});

$("clearBtn").addEventListener("click", () => {
  resetResult();
  $("uploadPreviewWrap").classList.add("hidden");
  $("capturePreviewWrap").classList.add("hidden");
  state.uploadedFile = null;
  state.capturedBlob = null;
  $("imageInput").value = "";
});

function addMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "W";
    wrapper.appendChild(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  $("messages").appendChild(wrapper);

  $("messages").scrollTop = $("messages").scrollHeight;
}

$("chatForm").addEventListener("submit", async event => {
  event.preventDefault();

  const input = $("chatInput");
  const message = input.value.trim();

  if (!message || state.busy) return;

  addMessage("user", message);
  input.value = "";

  state.busy = true;

  const typing = document.createElement("div");
  typing.className = "message assistant";
  typing.id = "typingMessage";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "W";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = "Thinking...";

  typing.appendChild(avatar);
  typing.appendChild(bubble);
  $("messages").appendChild(typing);
  $("messages").scrollTop = $("messages").scrollHeight;

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message })
    });

    const data = await parseApiResponse(response);

    typing.remove();
    addMessage("assistant", data.message || "Sorry, I couldn't generate a response.");
  } catch (error) {
    typing.remove();
    addMessage("assistant", `Sorry, something went wrong: ${error.message}`);
    showToast(error.message);
  } finally {
    state.busy = false;
  }
});

$("clearChatBtn").addEventListener("click", () => {
  $("messages").innerHTML = `
    <div class="message assistant">
      <div class="avatar">W</div>
      <div class="bubble">Hello! I am the WeAR AI. How can I help you find the perfect glasses frames today?</div>
    </div>
  `;
});

window.addEventListener("beforeunload", stopCamera);

$("year").textContent = new Date().getFullYear();

// Start on Webcam mode.
activateMode("webcam");
