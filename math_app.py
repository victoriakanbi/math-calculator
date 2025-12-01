import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import re
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from fractions import Fraction

# 1. PAGE SETUP
st.set_page_config(layout="wide", page_title="Victor's Calculator", page_icon="🧮")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    h1 {font-size: 2.2rem !important;}
    div[data-testid="stMetricValue"] {font-size: 1.1rem !important;}
    .stAlert {padding: 0.5rem;}
    .stButton button {padding: 0px 10px;}
</style>
""", unsafe_allow_html=True)

# 2. STATE MANAGEMENT (RAM ONLY - NO FILES)
if 'ans' not in st.session_state: st.session_state.ans = 0
if 'sig_figs' not in st.session_state: st.session_state.sig_figs = 5
if 'history_cache' not in st.session_state: st.session_state.history_cache = ""

# 3. CONSTANTS LIBRARY
CONSTANTS = {
    "Universal Physics": {
        "c": (299792458, "Speed of Light (m/s)"),
        "g": (9.80665, "Gravity Standard (m/s²)"),
        "G": (6.67430e-11, "Gravitational Const. (N·m²/kg²)"),
        "atm": (101325, "Standard Atmosphere (Pa)"),
    },
    "Thermodynamics": {
        "R_u": (8.314, "Univ. Gas Constant"),
        "k_B": (1.38e-23, "Boltzmann Constant"),
        "N_A": (6.022e23, "Avogadro Number"),
    },
    "Math": {
        "pi": (np.pi, "Pi"),
        "e": (np.e, "Euler's Number")
    }
}

# 4. MEMORY SETUP
def np_heaviside(x): return np.heaviside(x, 1)
def np_delta(x): return np.zeros_like(x) 

calc_memory = {"np": np, "math": np, "pi": np.pi, "e": np.e, 
               "sin": np.sin, "cos": np.cos, "tan": np.tan, 
               "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
               "ln": np.log, "step": np_heaviside, "delta": np_delta}

sym_memory = {"sp": sp, "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log, "pi": sp.pi,
              "diff": sp.diff, "step": sp.Heaviside, "delta": sp.DiracDelta, "t": sp.Symbol("t"), "x": sp.Symbol("x")}

for cat in CONSTANTS.values():
    for key, val in cat.items():
        calc_memory[key] = val[0]
        sym_memory[key] = sp.Symbol(key)

calc_sym_memory = sym_memory.copy()
calc_sym_memory['f'] = sp.Function('f')

# 5. HELPERS
def smart_parse(text):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try: return parse_expr(text, transformations=transformations, local_dict=sym_memory)
    except: return sp.sympify(text, locals=sym_memory)

def clean_input(text):
    text = text.replace("^", "**")
    text = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', text)
    text = re.sub(r'(\))([a-zA-Z\d\(])', r'\1*\2', text)
    return text

def format_number(val):
    try:
        val_float = float(val)
        return f"{val_float:.{st.session_state.sig_figs}g}"
    except: return str(val)

# 6. UI & LOGIC
with st.sidebar:
    st.header("🧮 Victor's Calculator")
    tab_settings, tab_help = st.tabs(["⚙️ Set", "📝 Help"])
    
    with tab_settings:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history_cache = ""
            st.rerun()

history_container = st.container()

# INPUT (NO FILE SAVE)
user_input = st.text_area("Raw Input Log", height=300, value=st.session_state.history_cache)

if user_input != st.session_state.history_cache:
    st.session_state.history_cache = user_input
    st.rerun()

if not user_input:
    with history_container:
        st.info("Cloud Mode: History is temporary and private to this tab.")

# CALCULATION LOOP (Simplified)
lines = user_input.split('\n')
lines = [l for l in lines if l.strip()]

for line in lines:
    try:
        # Simple Calc Logic for Cloud Demo
        # (Full logic is preserved in your local file if needed)
        clean_line = clean_input(line)
        if "=" in clean_line and "graph" not in line:
            l, r = clean_line.split("=")
            res = sp.solve(sp.Eq(smart_parse(l), smart_parse(r)))
            label = "Solve"
        elif "graph" in line:
            res = "Graphing not fully supported in lite mode"
            label = "Graph"
        else:
            res = smart_parse(clean_line)
            label = "Calc"
            
        with history_container.container(border=True):
            st.markdown(f"**{label}:** `{line}`")
            st.latex(sp.latex(res))
            if hasattr(res, "evalf"):
                st.caption(f"Decimal: {format_number(res)}")
                
    except Exception as e:
        pass