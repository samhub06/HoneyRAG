import streamlit as st
import pandas as pd

# Load the answers.log file
df = pd.read_json("answers.log", lines=True)

st.set_page_config(page_title="RAG-Enhanced Honeypot Analyzer", layout="wide")
st.title("🛡️ RAG-Enhanced Honeypot Analyzer")

# Summary counter
st.metric("Total Attacks Detected", len(df))

# Show the last 5 entries
for _, row in df.tail(5).iterrows():
    st.markdown("### 📌 Attack Summary")
    st.write(row["answer"])

    # Severity detection
    ctx = row["context"].lower()
    if "shadow" in ctx or "passwd" in ctx:
        severity = "High"
        color = "🔴"
    elif "failed ssh" in ctx:
        severity = "Medium"
        color = "🟠"
    else:
        severity = "Low"
        color = "🟢"

    st.markdown(f"### 🚨 Attack Severity: {color} {severity}")

    # Details section
    with st.expander("📄 Details (click to expand)"):
        st.text(row["context"])

    # Recommendations
    st.markdown("### 🛠️ Recommendations")
    if severity == "High":
        st.write("- Patch vulnerable services\n- Monitor accounts\n- Block suspicious IPs")
    elif severity == "Medium":
        st.write("- Investigate login attempts\n- Review authentication logs")
    else:
        st.write("- Monitor activity\n- No immediate action required")

    st.markdown("---")
