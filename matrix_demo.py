import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="10×10 Row-Zap Demo", layout="centered")

# -------------------------------------------------
# stato iniziale
# -------------------------------------------------
if "matrix" not in st.session_state:
    st.session_state.matrix = np.random.random((10, 10))
if "zapped" not in st.session_state:
    st.session_state.zapped = set()

# copia per il plot (non tocchiamo l'originale direttamente)
mat = st.session_state.matrix.copy()
for r in st.session_state.zapped:
    mat[r, :] = np.nan

# -------------------------------------------------
# plot
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(mat, aspect="auto", origin="upper")
ax.set_xlabel("Colonna")
ax.set_ylabel("Riga")

coords = streamlit_image_coordinates(fig, key="click")
st.pyplot(fig, clear_figure=True)

# -------------------------------------------------
# gestione click
# -------------------------------------------------
if coords is not None:
    y_pix = coords["y"]                       # pixel dall'alto
    nrows = mat.shape[0]
    row = int(y_pix / fig.bbox.height * nrows)
    st.session_state.zapped.add(row)
    st.experimental_rerun()

# -------------------------------------------------
# pannello info
# -------------------------------------------------
st.write("Righe zappate:", sorted(st.session_state.zapped) or "—")

col1, col2 = st.columns(2)
if col1.button("Nuova matrice"):
    st.session_state.matrix = np.random.random((10, 10))
    st.session_state.zapped = set()
    st.experimental_rerun()

if col2.button("Reset zapping"):
    st.session_state.zapped = set()
    st.experimental_rerun()
