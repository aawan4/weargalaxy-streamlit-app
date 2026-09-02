const API_BASE = window.location.origin;

const $ = (id) => document.getElementById(id);

const state = {
    mode: "webcam",
    stream: null,
    capturedBlob: null,
    uploadedFile: null,
    busy: false
};


/* ============================================================
   UI HELPERS
   ============================================================ */

function showToast(message) {

    const toast = $("toast");

    toast.textContent = message;
    toast.classList.add("show");

    clearTimeout(showToast.timer);

    showToast.timer = setTimeout(() => {
        toast.classList.remove("show");
    }, 4000);
}


function setLoading(loading) {

    state.busy = loading;

    $("loading").classList.toggle(
        "hidden",
        !loading
    );

    $("resultCard").classList.toggle(
        "hidden",
        loading
    );
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


/* ============================================================
   API RESPONSE
   ============================================================ */

async function apiRequest(
    endpoint,
    options = {}
) {

    const url =
        `${API_BASE}${endpoint}`;

    let response;

    try {

        response = await fetch(
            url,
            {
                ...options,
                headers: {
                    ...(options.headers || {})
                }
            }
        );

    } catch (error) {

        throw new Error(
            "Could not connect to the WeAR AI server."
        );
    }


    const contentType =
        response.headers.get(
            "content-type"
        ) || "";


    if (!contentType.includes(
        "application/json"
    )) {

        const text =
            await response.text();

        console.error(
            "Non-JSON API response:",
            text
        );

        throw new Error(
            `API returned HTTP ${response.status}.`
        );
    }


    const data =
        await response.json();


    if (!response.ok) {

        throw new Error(
            data.detail ||
            data.message ||
            `Server returned HTTP ${response.status}.`
        );
    }


    return data;
}


/* ============================================================
   MODE SWITCHING
   ============================================================ */

const tabs =
    document.querySelectorAll(
        ".mode-tab"
    );

const modeContents =
    document.querySelectorAll(
        ".mode-content"
    );


function activateMode(mode) {

    state.mode = mode;

    tabs.forEach(tab => {

        tab.classList.toggle(
            "active",
            tab.dataset.mode === mode
        );

    });


    modeContents.forEach(content => {

        content.classList.toggle(
            "active",
            content.id === `${mode}Mode`
        );

    });


    const isChat =
        mode === "chat";


    $("advisorSection")
        .classList
        .toggle(
            "hidden",
            isChat
        );


    $("chatSection")
        .classList
        .toggle(
            "hidden",
            !isChat
        );


    if (mode !== "webcam") {
        stopCamera();
    }
}


tabs.forEach(tab => {

    tab.addEventListener(
        "click",
        () => activateMode(
            tab.dataset.mode
        )
    );

});


/* ============================================================
   MOBILE MENU
   ============================================================ */

$("mobileMenu")
    ?.addEventListener(
        "click",
        () => {

            document
                .querySelector(
                    ".nav-links"
                )
                .classList
                .toggle("open");

        }
    );


/* ============================================================
   WEBCAM
   ============================================================ */

async function startCamera() {

    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {

        showToast(
            "Your browser does not support webcam access."
        );

        return;
    }


    try {

        stopCamera();

        state.stream =
            await navigator
                .mediaDevices
                .getUserMedia({

                    video: {
                        facingMode: "user",
                        width: {
                            ideal: 1280
                        },
                        height: {
                            ideal: 720
                        }
                    },

                    audio: false

                });


        const video =
            $("camera");

        video.srcObject =
            state.stream;

        video.classList.add(
            "ready"
        );

        $("cameraPlaceholder")
            .classList
            .add("hidden");

        $("captureBtn").disabled =
            false;

        showToast(
            "Camera started."
        );

    } catch (error) {

        console.error(error);

        showToast(
            "Camera access was blocked. Please allow camera permission."
        );
    }
}


function stopCamera() {

    if (state.stream) {

        state.stream
            .getTracks()
            .forEach(track => {
                track.stop();
            });

        state.stream = null;
    }


    const video =
        $("camera");

    if (video) {

        video.srcObject = null;

        video.classList.remove(
            "ready"
        );
    }


    $("cameraPlaceholder")
        ?.classList
        .remove("hidden");


    if ($("captureBtn")) {
        $("captureBtn").disabled = true;
    }
}


$("startCameraBtn")
    ?.addEventListener(
        "click",
        startCamera
    );


$("captureBtn")
    ?.addEventListener(
        "click",
        () => {

            const video =
                $("camera");


            if (
                !video.videoWidth ||
                !video.videoHeight
            ) {

                showToast(
                    "Camera is not ready yet."
                );

                return;
            }


            const canvas =
                $("cameraCanvas");


            canvas.width =
                video.videoWidth;

            canvas.height =
                video.videoHeight;


            const ctx =
                canvas.getContext(
                    "2d"
                );


            ctx.drawImage(
                video,
                0,
                0,
                canvas.width,
                canvas.height
            );


            canvas.toBlob(
                blob => {

                    if (!blob) {

                        showToast(
                            "Could not capture the photo."
                        );

                        return;
                    }


                    state.capturedBlob =
                        blob;


                    $("capturePreview")
                        .src =
                        URL.createObjectURL(
                            blob
                        );


                    $("capturePreviewWrap")
                        .classList
                        .remove(
                            "hidden"
                        );

                },
                "image/jpeg",
                0.9
            );

        }
    );


/* ============================================================
   WEBCAM ANALYSIS
   ============================================================ */

$("analyzeCaptureBtn")
    ?.addEventListener(
        "click",
        async () => {

            if (
                !state.capturedBlob
            ) {

                showToast(
                    "Capture a photo first."
                );

                return;
            }


            setLoading(true);


            try {

                const reader =
                    new FileReader();


                reader.onload =
                    async () => {

                        try {

                            const data =
                                await apiRequest(
                                    "/api/analyze-base64",
                                    {
                                        method: "POST",

                                        headers: {
                                            "Content-Type":
                                                "application/json"
                                        },

                                        body:
                                            JSON.stringify({
                                                image:
                                                    reader.result
                                            })
                                    }
                                );


                            showResult(
                                "Analysis Complete",
                                data.analysis ||
                                "No analysis was returned."
                            );

                        } catch (error) {

                            showResult(
                                "Analysis Failed",
                                error.message
                            );

                            showToast(
                                error.message
                            );
                        }

                    };


                reader.onerror =
                    () => {

                        setLoading(false);

                        showToast(
                            "Could not read captured image."
                        );
                    };


                reader.readAsDataURL(
                    state.capturedBlob
                );


            } catch (error) {

                showResult(
                    "Analysis Failed",
                    error.message
                );

                showToast(
                    error.message
                );
            }
        }
    );


/* ============================================================
   IMAGE UPLOAD
   ============================================================ */

$("imageInput")
    ?.addEventListener(
        "change",
        event => {

            const file =
                event.target.files?.[0];


            if (!file) {
                return;
            }


            const allowed = [
                "image/jpeg",
                "image/png",
                "image/webp"
            ];


            if (
                !allowed.includes(
                    file.type
                )
            ) {

                showToast(
                    "Please choose a JPG, PNG, or WEBP image."
                );

                event.target.value = "";

                return;
            }


            if (
                file.size >
                10 * 1024 * 1024
            ) {

                showToast(
                    "Image is larger than 10 MB."
                );

                event.target.value = "";

                return;
            }


            state.uploadedFile =
                file;


            $("uploadPreview")
                .src =
                URL.createObjectURL(
                    file
                );


            $("uploadPreviewWrap")
                .classList
                .remove(
                    "hidden"
                );

        }
    );


$("analyzeUploadBtn")
    ?.addEventListener(
        "click",
        async () => {

            if (
                !state.uploadedFile
            ) {

                showToast(
                    "Choose an image first."
                );

                return;
            }


            if (state.busy) {
                return;
            }


            setLoading(true);


            try {

                const form =
                    new FormData();


                form.append(
                    "file",
                    state.uploadedFile,
                    state.uploadedFile.name
                );


                const data =
                    await apiRequest(
                        "/api/analyze",
                        {
                            method: "POST",
                            body: form
                        }
                    );


                showResult(
                    "Analysis Complete",
                    data.analysis ||
                    "No analysis was returned."
                );


            } catch (error) {

                showResult(
                    "Analysis Failed",
                    error.message
                );

                showToast(
                    error.message
                );
            }
        }
    );


/* ============================================================
   MANUAL RECOMMENDATION
   ============================================================ */

$("manualBtn")
    ?.addEventListener(
        "click",
        async () => {

            const shape =
                $("shapeSelect").value;


            if (!shape) {

                showToast(
                    "Please select your face shape."
                );

                return;
            }


            if (state.busy) {
                return;
            }


            setLoading(true);


            try {

                console.log(
                    "Calling:",
                    `${API_BASE}/api/suggestion`
                );


                const data =
                    await apiRequest(
                        "/api/suggestion",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify({
                                    shape: shape
                                })
                        }
                    );


                console.log(
                    "Recommendation:",
                    data
                );


                showResult(
                    `Your Face Shape Is: ${data.face_shape || shape}`,
                    data.recommendation ||
                    "No recommendation was returned."
                );


            } catch (error) {

                console.error(
                    "Recommendation error:",
                    error
                );


                showResult(
                    "Recommendation Failed",
                    error.message
                );


                showToast(
                    error.message
                );
            }
        }
    );


/* ============================================================
   CLEAR RESULT
   ============================================================ */

$("clearBtn")
    ?.addEventListener(
        "click",
        () => {

            resetResult();

            $("uploadPreviewWrap")
                ?.classList
                .add("hidden");

            $("capturePreviewWrap")
                ?.classList
                .add("hidden");

            state.uploadedFile =
                null;

            state.capturedBlob =
                null;

            if ($("imageInput")) {
                $("imageInput").value =
                    "";
            }
        }
    );


/* ============================================================
   CHAT
   ============================================================ */

function addMessage(
    role,
    text
) {

    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        `message ${role}`;


    if (role === "assistant") {

        const avatar =
            document.createElement(
                "div"
            );

        avatar.className =
            "avatar";

        avatar.textContent =
            "W";

        wrapper.appendChild(
            avatar
        );
    }


    const bubble =
        document.createElement(
            "div"
        );

    bubble.className =
        "bubble";

    bubble.textContent =
        text;


    wrapper.appendChild(
        bubble
    );


    $("messages")
        .appendChild(
            wrapper
        );


    $("messages").scrollTop =
        $("messages").scrollHeight;
}


$("chatForm")
    ?.addEventListener(
        "submit",
        async event => {

            event.preventDefault();


            const input =
                $("chatInput");


            const message =
                input.value.trim();


            if (
                !message ||
                state.busy
            ) {
                return;
            }


            addMessage(
                "user",
                message
            );


            input.value = "";


            state.busy = true;


            try {

                const data =
                    await apiRequest(
                        "/api/chat",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify({
                                    message
                                })
                        }
                    );


                addMessage(
                    "assistant",
                    data.message ||
                    "I couldn't generate a response."
                );


            } catch (error) {

                addMessage(
                    "assistant",
                    `Sorry, something went wrong: ${error.message}`
                );

                showToast(
                    error.message
                );

            } finally {

                state.busy =
                    false;
            }

        }
    );


/* ============================================================
   CLEAR CHAT
   ============================================================ */

$("clearChatBtn")
    ?.addEventListener(
        "click",
        () => {

            $("messages").innerHTML = `
                <div class="message assistant">
                    <div class="avatar">W</div>
                    <div class="bubble">
                        Hello! I am the WeAR AI.
                        How can I help you find the perfect glasses frames today?
                    </div>
                </div>
            `;

        }
    );


/* ============================================================
   CLEANUP
   ============================================================ */

window.addEventListener(
    "beforeunload",
    stopCamera
);


$("year").textContent =
    new Date().getFullYear();


activateMode(
    "webcam"
);
