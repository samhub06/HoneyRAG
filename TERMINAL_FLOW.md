\# -------------------------------

\# 1. Start Honeypot (Terminal 1)

\# -------------------------------

cd \~/cowrie

source cowrie-env/bin/activate

pip install -e .

cowrie start



\# -------------------------------

\# 2. Simulate Attacker (Terminal 2)

\# -------------------------------

ssh -p 2222 root@127.0.0.1

\# Example attacker commands

whoami

uname -a

cat /etc/passwd

exit



\# -------------------------------

\# 3. Ingest Logs \& Build Index (Terminal 3)

\# -------------------------------

cd orchestrator

python3 ingest\_logs.py --source ../artifacts/demo\_logs.json --mode batch

python3 build\_index.py --input ../artifacts/demo\_logs.json --output ../artifacts/faiss\_index



\# -------------------------------

\# 4. Query Index (Terminal 3)

\# -------------------------------

python3 query\_index.py "failed ssh login" 10 \\

"Summarize suspicious activity and suggest two next steps."



\# -------------------------------

\# 5. Launch Dashboard (Terminal 4)

\# -------------------------------

cd ui

streamlit run dashboard.py



