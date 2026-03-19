"""
Streamlit wrapper for Hedge Fund Radar Pro.
Serves the static HTML dashboard when deployed to Streamlit Cloud.
For GitHub Pages, use index.html directly.
"""
import streamlit as st
import subprocess, os, json

st.set_page_config(page_title="Ultra Scanner Pro", page_icon="🎯", layout="wide")

# Build data if not present
if not os.path.exists("data/snapshot.json"):
    with st.spinner("Building scanner data (first run, ~5 min)..."):
        subprocess.run(["python", "scripts/build_data.py", "--out-dir", "data"], check=True)

# Read and serve HTML with embedded data
with open("index.html", "r") as f:
    html = f.read()

# Inject data inline for Streamlit (can't serve static files)
if os.path.exists("data/snapshot.json"):
    with open("data/snapshot.json", "r") as f:
        data = f.read()
    # Replace the fetch() call with inline data
    inject = f"""<script>
    window._INLINE_DATA = {data};
    </script>"""
    html = html.replace("</head>", inject + "\n</head>")
    # Replace the fetch call in JS
    html = html.replace(
        "fetch('data/snapshot.json').then(function(r){return r.ok?r.json():null}).then(function(d){",
        "Promise.resolve(window._INLINE_DATA).then(function(d){"
    )

st.components.v1.html(html, height=900, scrolling=True)
