// basic DOM helpers
function el(q) { return document.querySelector(q) }
function elAll(q) { return document.querySelectorAll(q) }

const chatbox = el('#chatbox');
const input = el('#textInput');
const sendBtn = el('#sendBtn');
const micBtn = el('#micBtn');
const ttsBtn = el('#ttsBtn');
const loginBtn = el('#loginBtn');
const loginModal = el('#loginModal');
const closeModal = el('#closeModal');
const doLogin = el('#doLogin');
const toggleTheme = el('#toggleTheme');

let lastBotText = "";
let recognition = null;
let isDark = true;

// send quick helper
function sendQuick(text) { input.value = text; sendMessage(); }

// scroll
function scrollDown() { chatbox.scrollTop = chatbox.scrollHeight; }

// add messages
function addUserMessage(text) {
    chatbox.insertAdjacentHTML('beforeend',
        `<div class="message user-msg"><div class="bubble user-bubble">${escapeHtml(text)}</div></div>`);
    scrollDown();
}
function addBotMessage(text) {
    lastBotText = text;
    chatbox.insertAdjacentHTML('beforeend',
        `<div class="message bot-msg"><div class="bubble bot-bubble">${escapeHtml(text)}</div></div>`);
    scrollDown();
}

// escape
function escapeHtml(unsafe) { return unsafe.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'); }

// typing indicator
function addTyping() {
    chatbox.insertAdjacentHTML('beforeend',
        `<div class="message bot-msg" id="typing"><div class="bubble bot-bubble">Ghost is typing<span class="dots">...</span></div></div>`);
    scrollDown();
}
function removeTyping() { let t = document.getElementById('typing'); if (t) t.remove(); }

// send
function sendMessage() {
    const text = input.value.trim();
    if (!text) return;
    addUserMessage(text);
    input.value = "";
    addTyping();

    fetch(`/get?msg=${encodeURIComponent(text)}`)
        .then(r => r.json())
        .then(data => {
            removeTyping();
            addBotMessage(data);
        })
        .catch(err => {
            removeTyping();
            addBotMessage("Sorry — there was a network error.");
            console.error(err);
        });
}

// keyboard
input.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });
sendBtn.addEventListener('click', sendMessage);

// mic (SpeechRecognition)
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.onresult = e => {
        input.value = e.results[0][0].transcript;
        sendMessage();
    };
    recognition.onerror = e => console.warn('Speech recognition error', e);
    micBtn.addEventListener('click', () => { if (recognition) recognition.start(); });
} else { micBtn.style.display = 'none'; }

// TTS
function speak(text) {
    if (!('speechSynthesis' in window)) return;
    const ut = new SpeechSynthesisUtterance(text);
    ut.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(ut);
}
ttsBtn.addEventListener('click', () => { if (lastBotText) speak(lastBotText); });

// login modal
loginBtn.addEventListener('click', () => { loginModal.style.display = 'flex'; });
closeModal.addEventListener('click', () => { loginModal.style.display = 'none'; });
doLogin.addEventListener('click', () => {
    const u = el('#username').value, p = el('#password').value;
    fetch('/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: u, password: p }) })
        .then(r => r.json()).then(res => {
            if (res.success) { addBotMessage('Logged in as ' + res.user); loginModal.style.display = 'none'; }
            else addBotMessage('Login failed. Demo credentials: user / pass');
        });
});

// theme toggle (very simple)
toggleTheme.addEventListener('click', () => {
    isDark = !isDark;
    document.documentElement.style.setProperty('--bg', isDark ? '#0f1720' : '#f4f7f6');
    document.documentElement.style.setProperty('--card', isDark ? '#0b1220' : '#ffffff');
    document.body.style.background = isDark ? 'var(--bg)' : '#f4f7f6';
});

// helper to call quick actions from sidebar
function sendQuickMessage(text) { input.value = text; sendMessage(); }