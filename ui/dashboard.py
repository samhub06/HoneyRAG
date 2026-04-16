import streamlit as st
import subprocess

st.title("HoneyRAG Dashboard")

if st.button("Run demo pipeline"):
    with st.spinner("Running pipeline..."):
        proc = subprocess.run(["python", "orchestrator/flow_demo.py"], capture_output=True, text=True)
        st.code(proc.stdout)
        if proc.stderr:
            st.error(proc.stderr)

st.info("Connect Prefect Orion for full run history.")