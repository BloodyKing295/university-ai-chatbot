from flask import Flask, render_template_string, request, jsonify
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import ollama
import math

app = Flask(__name__)

print("Loading embedding model...")
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Reading university PDF...")
reader = PdfReader("university.pdf")
pdf_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n"

if not pdf_text.strip():
    raise ValueError("university.pdf has no readable text.")

chunk_size = 500
chunks = [
    pdf_text[i:i + chunk_size]
    for i in range(0, len(pdf_text), chunk_size)
]

print("Creating embeddings...")
chunk_embeddings = embedding_model.encode(chunks)
print("Chatbot backend ready!")


def cosine_similarity(a, b):
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    mag_a = math.sqrt(sum(float(x) ** 2 for x in a))
    mag_b = math.sqrt(sum(float(y) ** 2 for y in b))

    if mag_a == 0 or mag_b == 0:
        return 0

    return dot / (mag_a * mag_b)


def retrieve_chunks(question, top_k=2):
    question_embedding = embedding_model.encode([question])[0]
    scores = []

    for index, embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, embedding)
        scores.append((score, chunks[index]))

    scores.sort(key=lambda item: item[0], reverse=True)

    if not scores or scores[0][0] < 0.25:
        return []

    return [chunk for score, chunk in scores[:top_k]]


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>University AI Assistant</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f1f5f9;
    color: #172033;
}

.layout {
    display: flex;
    max-width: 1100px;
    min-height: 90vh;
    margin: 25px auto;
    gap: 18px;
    padding: 15px;
}

.sidebar {
    width: 250px;
    flex-shrink: 0;
    background: white;
    padding: 18px;
    border-radius: 14px;
}

.sidebar h3 {
    margin-top: 0;
}

button {
    border: none;
    padding: 11px 14px;
    border-radius: 9px;
    background: #2563eb;
    color: white;
    cursor: pointer;
}

button:hover {
    opacity: 0.9;
}

.new-chat {
    width: 100%;
    margin-bottom: 20px;
}

.history-item {
    display: block;
    width: 100%;
    text-align: left;
    background: #f1f5f9;
    color: #172033;
    margin: 8px 0;
    overflow-wrap: anywhere;
}

.history-item.active {
    background: #dbeafe;
}

.main {
    flex: 1;
    min-width: 0;
}

h1 {
    margin: 0 0 8px;
}

.subtitle {
    color: #64748b;
    margin-bottom: 20px;
}

.chat {
    background: white;
    border-radius: 14px;
    padding: 18px;
    min-height: 350px;
    max-height: 55vh;
    overflow-y: auto;
    box-shadow: 0 4px 18px #0000000d;
}

.message {
    padding: 12px 14px;
    margin: 10px 0;
    border-radius: 12px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.user {
    background: #dbeafe;
    margin-left: 30px;
}

.bot {
    background: #f1f5f9;
    margin-right: 20px;
}

form {
    display: flex;
    gap: 8px;
    margin-top: 15px;
}

input {
    flex: 1;
    min-width: 0;
    padding: 13px;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    font-size: 16px;
}

.status {
    color: #64748b;
    font-size: 13px;
    margin-top: 12px;
}

@media (max-width: 650px) {
    .layout {
        flex-direction: column;
        margin: 0;
        padding: 12px;
    }

    .sidebar {
        width: 100%;
    }

    .chat {
        min-height: 280px;
    }
}
</style>
</head>

<body>
<div class="layout">

    <aside class="sidebar">
        <h3>💬 Chat History</h3>

        <button class="new-chat" id="newChat">
            + New Chat
        </button>

        <div id="history"></div>
    </aside>

    <main class="main">
        <h1>🎓 University AI Assistant</h1>

        <div class="subtitle">
            Ask questions about university information.
        </div>

        <div class="chat" id="chat"></div>

        <form id="form">
            <input
                id="question"
                placeholder="Ask a question..."
                required
            >
            <button id="send" type="submit">Send</button>
        </form>

        <div class="status" id="status">
            Local AI • University PDF
        </div>
    </main>

</div>

<script>
const form = document.getElementById("form");
const input = document.getElementById("question");
const chat = document.getElementById("chat");
const historyBox = document.getElementById("history");
const statusText = document.getElementById("status");
const sendButton = document.getElementById("send");
const newChatButton = document.getElementById("newChat");

let conversations = [];
let currentId = null;
let nextId = 1;
let busy = false;

function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = "message " + who;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function getCurrentConversation() {
    return conversations.find(c => c.id === currentId);
}

function renderHistory() {
    historyBox.replaceChildren();

    conversations.forEach(conversation => {
        const button = document.createElement("button");
        button.className = "history-item";

        if (conversation.id === currentId) {
            button.classList.add("active");
        }

        button.textContent = conversation.title;

        button.addEventListener("click", () => {
            if (busy) return;
            currentId = conversation.id;
            renderConversation();
        });

        historyBox.appendChild(button);
    });
}

function renderConversation() {
    chat.replaceChildren();

    const conversation = getCurrentConversation();

    if (conversation) {
        conversation.messages.forEach(message => {
            addMessage(message.text, message.who);
        });
    }

    renderHistory();
}

function createNewChat() {
    if (busy) return;

    const conversation = {
        id: nextId++,
        title: "New conversation",
        messages: [
            {
                text: "Hello! Ask me a question about the university.",
                who: "bot"
            }
        ]
    };

    conversations.unshift(conversation);
    currentId = conversation.id;

    renderConversation();
    statusText.textContent = "Local AI • University PDF";
    input.focus();
}

newChatButton.addEventListener("click", createNewChat);

form.addEventListener("submit", async event => {
    event.preventDefault();

    const question = input.value.trim();

    if (!question || busy) return;

    let conversation = getCurrentConversation();

    if (!conversation) {
        createNewChat();
        conversation = getCurrentConversation();
    }

    conversation.messages.push({
        text: question,
        who: "user"
    });

    if (conversation.title === "New conversation") {
        conversation.title = question.length > 28
            ? question.slice(0, 28) + "..."
            : question;
    }

    renderConversation();

    input.value = "";
    busy = true;
    sendButton.disabled = true;
    newChatButton.disabled = true;
    statusText.textContent = "Thinking...";

    const conversationId = currentId;

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Request failed");
        }

        conversation.messages.push({
            text: data.answer,
            who: "bot"
        });

        statusText.textContent = "Local AI • University PDF";

    } catch (error) {
        conversation.messages.push({
            text: "Error: " + error.message,
            who: "bot"
        });

        statusText.textContent = "Please check the terminal.";

    } finally {
        busy = false;
        sendButton.disabled = false;
        newChatButton.disabled = false;

        if (currentId === conversationId) {
            renderConversation();
        } else {
            renderHistory();
        }

        input.focus();
    }
});

createNewChat();
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

    try:
        relevant_chunks = retrieve_chunks(question)

        if not relevant_chunks:
            return jsonify({
                "answer": "I don't have that information in the PDF."
            })

        context = "\n\n".join(relevant_chunks)

        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful university assistant. "
                        "Answer briefly using only the supplied PDF context. "
                        "Do not invent facts. If the answer is not in the "
                        "context, say: I don't have that information in the PDF."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"PDF context:\n{context}\n\nQuestion: {question}"
                    )
                }
            ],
            options={"temperature": 0.1}
        )

        answer = response["message"]["content"]
        return jsonify({"answer": answer})

    except Exception as error:
        print("Chatbot error:", error)
        return jsonify({
            "error": "Local AI could not answer. Check Ollama and terminal."
        }), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)