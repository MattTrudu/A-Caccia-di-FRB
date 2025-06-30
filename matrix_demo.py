import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import io

st.set_page_config(page_title="10×10 Row-Zap Demo", layout="centered")

# ---------------- stato -----------------
if "matrix" not in st.session_state:
    st.session_state.matrix = np.random.random((10, 10))
if "zapped" not in st.session_state:
    st.session_state.zapped = set()

mat = st.session_state.matrix.copy()
for r in st.session_state.zapped:
    mat[r, :] = np.nan

# ----------- crea la figura -------------
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(mat, aspect="auto", origin="upper")
ax.set_xlabel("Colonna")
ax.set_ylabel("Riga")
ax.set_title("Clicca per zappare una riga")

# ---- convertila in PNG in memoria ------
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
plt.close(fig)                # evita doppio output in Streamlit
buf.seek(0)
img = Image.open(buf)

# --------------- UI ---------------------
coords = streamlit_image_coordinates(img, key="click")
st.image(img)

# ---------- gestisci il click ----------
if coords is not None:
    row = int(coords["y"] / img.height * mat.shape[0])
    st.session_state.zapped.add(row)
    st.experimental_rerun()

# ----------- pulsanti utili -------------
st.write("Righe zappate:", sorted(st.session_state.zapped) or "—")
col1, col2 = st.columns(2)
if col1.button("Nuova matrice"):
    st.session_state.matrix = np.random.random((10, 10))
    st.session_state.zapped = set()
    st.experimental_rerun()

if col2.button("Reset zapping"):
    st.session_state.zapped = set()
    st.experimental_rerun()
