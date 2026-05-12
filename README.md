# HoneyRAG 🛡️🤖
**AI‑powered Honeypot with Retrieval‑Augmented Generation (RAG) for Intelligent Threat Analysis**

---

## 📌 Overview
HoneyRAG is a cybersecurity project that integrates **honeypot intrusion detection** with **Generative AI**. It captures attacker activity using Cowrie and other honeypots, then applies **Retrieval‑Augmented Generation (RAG)** to analyze logs, retrieve relevant threat intelligence, and generate contextual insights in real time.

---

## ❗ Problem Statement
Traditional honeypots are effective at logging attacker behavior but lack intelligent analysis. Security teams often face:
- Large volumes of raw logs with minimal context  
- Slow manual triage of incidents  
- Difficulty mapping attacker actions to known vulnerabilities  

---

## 💡 Solution
HoneyRAG bridges this gap by combining:
- **Cowrie Honeypot** → Captures attacker commands and behaviors  
- **Vector Database (FAISS)** → Stores embeddings of attacker logs for fast similarity search  
- **Sentence Transformers** → Converts logs into embeddings that capture semantic meaning  
- **Local LLM (GPT4All)** → Generates summaries, severity ratings, and CVE/MITRE mappings  
- **Prefect Orchestration** → Manages the pipeline stages end‑to‑end  
- **Streamlit Dashboard** → Provides an interactive UI for monitoring and analysis  

---

## 🔄 Workflow

### 1. Data Capture
- Honeypots (Cowrie, Dionaea) simulate vulnerable systems.  
- Attacker activity (commands, payloads, behaviors) is logged in real time.  

### 2. Preprocessing & Embedding
- Logs are cleaned and normalized.  
- Sentence Transformers generate embeddings for each log entry.  

### 3. Vector Index & Storage
- Embeddings stored in FAISS for millisecond retrieval.  
- Raw logs archived for historical analysis.  

### 4. Retrieval
- New activity triggers similarity search in FAISS.  
- Relevant past incidents and threat intelligence are retrieved.  

### 5. LLM Analysis
- Context passed into GPT4All.  
- Outputs include summaries, severity ratings, CVE/MITRE mappings, and response suggestions.  

### 6. Workflow Orchestration
- Prefect orchestrates the pipeline:  
  `Capture → Preprocess → Embed → Store → Retrieve → Analyze → Output`  

### 7. Dashboard & UI
- Streamlit dashboard displays:  
  - Attacker activity timeline  
  - AI‑generated summaries  
  - Threat intelligence matches  
  - Response recommendations  

### 8. Human Approval
- Analysts review AI suggestions.  
- Approved actions can trigger automated playbooks (e.g., block IP, alert SOC).  

---

## 🚀 Key Features
- Real‑time attacker activity logging  
- AI‑driven threat summarization and contextual insights  
- Knowledge base retrieval for faster incident response  
- Modular design for easy extension and integration  
- Human‑in‑the‑loop decision making  

---

## 📊 Impact
HoneyRAG demonstrates how **Generative AI can enhance traditional cybersecurity defenses**. Recruiters and collaborators can see:
- Practical coding achievement (Python, FAISS, Cowrie, Streamlit, Prefect)  
- End‑to‑end project completion with modular design  
- Application of AI/ML in a real‑world security context  

---

## 📷 Future Enhancements
- Integration with external threat intelligence feeds  
- Advanced visualization of attacker behavior  
- Cloud deployment for scalability  

---

## 👨‍💻 Author
**Samprakash (Sam)**  
- MCA (Generative AI specialization), SRM Institute of Science and Technology  
- GitHub: [github.com/samhub06](https://github.com/samhub06)  
- LinkedIn: [linkedin.com/in/samprakash-s-81aa76350](https://linkedin.com/in/samprakash-s-81aa76350)  
