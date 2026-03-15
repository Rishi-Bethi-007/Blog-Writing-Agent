# ✍️ Blog Writing Agent

An AI agent that researches, outlines, and writes full blog posts on any topic — with a live web interface.

🚀 **[Live Demo](https://blog-writing-agent-anzebhqvjlnwbkzmvg4amn.streamlit.app/)**

---

## What It Does

Give it a topic. The agent:
1. **Researches** the topic using web search
2. **Plans** a structure (title, sections, key points)
3. **Writes** each section with coherent flow
4. **Reviews** and refines the output for quality

The result is a complete, publication-ready blog post — not just a summary.

---

## 🏗️ Architecture
```
User Input (topic + tone + length preferences)
↓
Research Agent (web search + source gathering)
↓
Outline Planner (structure generation)
↓
Writing Agent (section-by-section generation)
↓
Editor Agent (review + refinement)
↓
Final Blog Post (Streamlit UI)
```

---

## ✨ Features

- 🔍 **Web-grounded research** — not hallucinated knowledge
- 🗂️ **Multi-step agent pipeline** — each stage is independently controllable
- ✏️ **Tone control** — professional, casual, technical
- 📏 **Length control** — short-form to long-form
- 🖥️ **Live Streamlit UI** — try it without running any code

---

## 🛠️ Tech Stack

**Python · LangGraph · OpenAI API · Streamlit · Tavily for search**

---

## 🚀 Run Locally
```bash
git clone https://github.com/Rishi-Bethi-007/Blog-Writing-Agent
cd Blog-Writing-Agent
pip install -r requirements.txt
# Add your API keys to .env
streamlit run app.py
```

---

## 👨‍💻 Author

**Rishi Bethi** — MSc AI & Automation, University West, Sweden

[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub Portfolio](https://github.com/Rishi-Bethi-007)
