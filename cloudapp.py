import os
import requests
from flask import Flask, render_template_string, request, jsonify
from pypdf import PdfReader

app = Flask(_name_)

# Read the Groq API key securely from an environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Read university information from the PDF
PDF_PATH = "university.pdf"
pdf_text = ""

if os.path.exists(PDF_PATH):
    reader = PdfReader(PDF_PATH)

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text + "\n"

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>University AI Assistant</title>
<style>
body {
    margin: 0;
    padding: 20px;
    font-family: Arial, sans-serif;
    background: #f1f5f9;
    color: #172033;
}
.container {
    max-width: 750px;
    margin: auto;
}
h1 { text-align: center; }
.chat {
    background: white;
    border-radius: 12px;
    padding: 15px;
    height: 55vh;
    overflow-y: auto;
}
.message {
    padding: 12px;
    margin: 10px 0;
    border-radius: 10px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.user { background: #dbeafe; }
.bot { background: #e2e8f0; }
form {
    display: flex;
    gap: 8px;
    margin-top: 12px;
}
input {
    flex: 1;
    min-width: 0;
    padding: 12px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    font-size: 16px;
}
button {
    padding: 12px 18px;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
}
#status {
    margin-top: 10px;
    color: #64748b;
    font-size: 13px;
}
</style>
</head>
<body>
<div class="container">
    <h1>University AI Assistant 🎓</h1>
    <p>Ask questions about university information.</p>

    <div class="chat" id="chat"></div>

    <form id="form">
        <input id="question" placeholder="Ask a question..." required>
        <button id="send" type="submit">Send</button>
    </form>

    <button id="newChat" type="button" style="margin-top:10px">
        New Chat
    </button>

    <div id="status">University AI Assistant</div>
</div>

<script>
const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("question");
const send = document.getElementById("send");
const statusText = document.getElementById("status");

function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = "message " + who;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function newChat() {
    chat.replaceChildren();
    addMessage(
        "Hello! Ask me about university office timings and other information.",
        "bot"
    );
    statusText.textContent = "Ready";
}

document.getElementById("newChat").addEventListener("click", newChat);

form.addEventListener("submit", async event => {
    event.preventDefault();

    const question = input.value.trim();
    if (!question) return;

    addMessage(question, "user");
    input.value = "";
    send.disabled = true;
    statusText.textContent = "Thinking...";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question})
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Request failed");
        }

        addMessage(data.answer, "bot");
        statusText.textContent = "Ready";

    } catch (error) {
        addMessage("Error: " + error.message, "bot");
        statusText.textContent = "Please try again.";
    } finally {
        send.disabled = false;
        input.focus();
    }
});

newChat();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if not GROQ_API_KEY:
        return jsonify({
            "error": "Server API key is not configured."
        }), 500

    if not pdf_text.strip():
        return jsonify({
            "answer": (
                "University information is not available yet. "
                "Please add the university PDF to the server."
            )
        })

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful university assistant. "
                            "Answer using only the university PDF context. "
                            "Do not invent office timings, dates, fees, "
                            "or other facts. If the answer is not in the "
                            "context, say you do not have that information."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "University PDF context:\n"
                            + pdf_text[:18000]
                            + "\n\nQuestion: "
                            + question
                        )
                    }
                ],
                "temperature": 0.1
            },
            timeout=60
        )

        if response.status_code != 200:
            print("Groq API error:", response.status_code, response.text)
            return jsonify({
                "error": "The AI service could not answer. Please try again."
            }), 502

        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        return jsonify({"answer": answer})

    except requests.RequestException as error:
        print("Network error:", error)
        return jsonify({
            "error": "Could not connect to the AI service."
        }), 502

    except (KeyError, IndexError, ValueError) as error:
        print("Response error:", error)
        return jsonify({
            "error": "The AI service returned an unexpected response."
        }), 502


if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)
