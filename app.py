from pathlib import Path
import sys

from flask import Flask, request, render_template_string

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from summarize import summarize_text

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SummarAI — AI Text Summarizer</title>
    <style>
        * { box-sizing: border-box; }

        :root {
            --bg: #080b12;
            --panel: rgba(17, 24, 39, 0.78);
            --panel-2: rgba(255,255,255,0.045);
            --border: rgba(255,255,255,0.10);
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #8b5cf6;
            --accent-2: #06b6d4;
            --success: #34d399;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                         "SF Pro Text", Inter, Segoe UI, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 15% 15%, rgba(139,92,246,.22), transparent 30%),
                radial-gradient(circle at 85% 20%, rgba(6,182,212,.16), transparent 28%),
                radial-gradient(circle at 50% 100%, rgba(99,102,241,.10), transparent 35%),
                var(--bg);
        }

        .orb {
            position: fixed;
            width: 280px;
            height: 280px;
            border-radius: 50%;
            filter: blur(90px);
            opacity: .22;
            pointer-events: none;
        }

        .orb.one { background: #7c3aed; top: -120px; left: -100px; }
        .orb.two { background: #0891b2; right: -100px; top: 180px; }

        .wrap {
            width: min(1180px, calc(100% - 36px));
            margin: 0 auto;
            padding: 34px 0 55px;
        }

        .nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 70px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 11px;
            font-weight: 750;
            letter-spacing: -.4px;
            font-size: 19px;
        }

        .logo {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: linear-gradient(135deg, #8b5cf6, #06b6d4);
            box-shadow: 0 8px 30px rgba(99,102,241,.32);
            font-size: 19px;
        }

        .badge {
            padding: 8px 13px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: rgba(255,255,255,.035);
            color: #cbd5e1;
            font-size: 12px;
            backdrop-filter: blur(15px);
        }

        .hero {
            text-align: center;
            max-width: 850px;
            margin: 0 auto 42px;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 12px;
            border: 1px solid rgba(139,92,246,.25);
            background: rgba(139,92,246,.08);
            color: #c4b5fd;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 650;
            margin-bottom: 18px;
        }

        .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 12px var(--success);
        }

        h1 {
            margin: 0;
            font-size: clamp(42px, 7vw, 76px);
            line-height: .98;
            letter-spacing: -4px;
            font-weight: 800;
            background: linear-gradient(100deg, #fff 20%, #c4b5fd 55%, #67e8f9);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .hero p {
            color: var(--muted);
            font-size: 17px;
            line-height: 1.65;
            max-width: 650px;
            margin: 22px auto 0;
        }

        .workspace {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            padding: 14px;
            border: 1px solid var(--border);
            border-radius: 26px;
            background: rgba(8,11,18,.66);
            box-shadow: 0 35px 100px rgba(0,0,0,.35);
            backdrop-filter: blur(25px);
        }

        .card {
            min-width: 0;
            border: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.025));
            border-radius: 20px;
            overflow: hidden;
        }

        .card-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 20px;
            border-bottom: 1px solid var(--border);
        }

        .card-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 700;
        }

        .icon {
            width: 31px;
            height: 31px;
            border-radius: 9px;
            display: grid;
            place-items: center;
            background: rgba(139,92,246,.13);
            color: #c4b5fd;
        }

        .counter {
            color: #64748b;
            font-size: 12px;
        }

        textarea {
            width: 100%;
            height: 360px;
            resize: vertical;
            border: 0;
            outline: none;
            padding: 22px;
            color: #e2e8f0;
            background: transparent;
            font: inherit;
            font-size: 15px;
            line-height: 1.75;
        }

        textarea::placeholder { color: #64748b; }

        .card-foot {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 14px 16px;
            border-top: 1px solid var(--border);
        }

        .hint {
            color: #64748b;
            font-size: 12px;
        }

        button {
            border: 0;
            color: white;
            cursor: pointer;
            padding: 12px 19px;
            border-radius: 11px;
            font: inherit;
            font-size: 13px;
            font-weight: 700;
            background: linear-gradient(135deg, #7c3aed, #0891b2);
            box-shadow: 0 8px 25px rgba(99,102,241,.25);
            transition: transform .18s ease, box-shadow .18s ease, opacity .18s ease;
        }

        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 32px rgba(99,102,241,.35);
        }

        button:active { transform: translateY(0); }

        .summary {
            min-height: 360px;
            padding: 25px 22px;
            color: #dbeafe;
            line-height: 1.8;
            font-size: 15px;
        }

        .summary.empty {
            display: grid;
            place-items: center;
            text-align: center;
            color: #64748b;
        }

        .empty-inner {
            max-width: 260px;
        }

        .empty-icon {
            width: 55px;
            height: 55px;
            margin: 0 auto 15px;
            display: grid;
            place-items: center;
            border-radius: 17px;
            background: rgba(6,182,212,.09);
            color: #67e8f9;
            font-size: 22px;
        }

        .summary p { margin: 0; }

        .error {
            margin: 16px 0 0;
            padding: 13px 15px;
            border-radius: 12px;
            border: 1px solid rgba(248,113,113,.25);
            background: rgba(127,29,29,.18);
            color: #fca5a5;
            font-size: 13px;
        }

        .features {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 17px;
        }

        .feature {
            padding: 17px;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(255,255,255,.025);
        }

        .feature strong {
            display: block;
            font-size: 13px;
            margin-bottom: 5px;
        }

        .feature span {
            color: #64748b;
            font-size: 12px;
        }

        footer {
            text-align: center;
            color: #475569;
            font-size: 12px;
            margin-top: 30px;
        }

        @media (max-width: 800px) {
            .nav { margin-bottom: 45px; }
            .workspace { grid-template-columns: 1fr; }
            .features { grid-template-columns: 1fr; }
            h1 { letter-spacing: -2.5px; }
        }
    </style>
</head>

<body>
    <div class="orb one"></div>
    <div class="orb two"></div>

    <main class="wrap">
        <nav class="nav">
            <div class="brand">
                <div class="logo">✦</div>
                SummarAI
            </div>
            <div class="badge">Transformer • Abstractive NLP</div>
        </nav>

        <section class="hero">
            <div class="eyebrow"><span class="dot"></span> AI-powered summarization</div>
            <h1>Turn long text into clear ideas.</h1>
            <p>
                Paste an article, report, or document and let a Transformer model
                extract the most important information into a concise summary.
            </p>
        </section>

        <form method="POST">
            <section class="workspace">

                <div class="card">
                    <div class="card-head">
                        <div class="card-title">
                            <div class="icon">↗</div>
                            Your document
                        </div>
                        <span class="counter" id="counter">0 characters</span>
                    </div>

                    <textarea
                        id="text"
                        name="text"
                        maxlength="12000"
                        placeholder="Paste your article or document here..."
                        required>{{ text }}</textarea>

                    <div class="card-foot">
                        <span class="hint">Best results with detailed text</span>
                        <button type="submit">Generate Summary&nbsp; →</button>
                    </div>
                </div>

                <div class="card">
                    <div class="card-head">
                        <div class="card-title">
                            <div class="icon">✦</div>
                            AI summary
                        </div>
                        <span class="counter">Transformer output</span>
                    </div>

                    {% if summary %}
                    <div class="summary">
                        <p>{{ summary }}</p>
                    </div>
                    {% else %}
                    <div class="summary empty">
                        <div class="empty-inner">
                            <div class="empty-icon">✦</div>
                            <strong>Your summary will appear here</strong>
                            <div style="margin-top:7px;font-size:12px;">
                                Your generated result will be shown in this panel.
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>

            </section>
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        <section class="features">
            <div class="feature">
                <strong>Abstractive AI</strong>
                <span>Generates new concise sentences, not just copied text.</span>
            </div>
            <div class="feature">
                <strong>Transformer model</strong>
                <span>Uses attention to understand relationships in the document.</span>
            </div>
            <div class="feature">
                <strong>Fast demo</strong>
                <span>Runs locally through your Flask application.</span>
            </div>
        </section>

        <footer>Text Summarization Project • Deep Learning & NLP</footer>
    </main>

    <script>
        const text = document.getElementById("text");
        const counter = document.getElementById("counter");

        function updateCounter() {
            const n = text.value.length;
            counter.textContent = n.toLocaleString() + " characters";
        }

        text.addEventListener("input", updateCounter);
        updateCounter();
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    summary = ""
    error = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            try:
                summary = summarize_text(text)
            except Exception as exc:
                error = f"Unable to generate summary: {exc}"
        else:
            error = "Please enter some text first."

    return render_template_string(
        HTML,
        text=text,
        summary=summary,
        error=error,
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
