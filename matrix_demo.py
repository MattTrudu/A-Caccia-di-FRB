# matrix_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import io

st.set_page_config(page_title="10×10 Row-Zap", layout="centered")

# ---------- stato ----------
if "matrix" not in st.session_state:
    st.session_state.matrix = np.random.random((10, 10))
if "zapped" not in st.session_state:
    st.session_state.zapped = set()

mat = st.session_state.matrix.copy()
for r in st.session_state.zapped:
    mat[r, :] = np.nan

# ---------- figura ----------
fig, ax = plt.subplots(figsize=(4, 4), facecolor="white")
ax.imshow(mat, aspect="auto", origin="upper")
ax.set_xlabel("Colonna")
ax.set_ylabel("Riga")
ax.set_title("Clicca per zappare una riga")
fig.tight_layout()

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")   # niente transparent
plt.close(fig)
buf.seek(0)
img = Image.open(buf)

# una colonna per l'immagine, una per i pulsanti
col_img, col_ctrl = st.columns([1, 1])

with col_img:
    coords = streamlit_image_coordinates(img, key="click", width=350)

with col_ctrl:
    st.markdown("#### Righe zappate")
    st.write(sorted(st.session_state.zapped) or "—")

    if st.button("Nuova matrice"):
        st.session_state.matrix = n_
