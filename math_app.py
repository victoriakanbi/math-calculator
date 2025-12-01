import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import os
import string
import re
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from fractions import Fraction

# 1. PAGE SETUP
st.set_page_config(
    layout="wide", 
    page_title="Victor's Calculator",
    page_icon="🧮"
)

# --- CUSTOM CSS FOR UI POLISH ---
st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    h1 {font-size: 2.2rem !important;}
    div[data-testid="stMetricValue"] {font-size: 1.1rem !important;}
    .stAlert {padding: 0.5rem;}
    .stButton button {padding: 0px 10px;}
</style>
""", unsafe_allow_html=True)

# --- 2. STATE MANAGEMENT (RAM ONLY) ---
if 'ans' not in st.session_state: st.session_state.ans = 0
if 'sig_figs' not in st.session_state: st.session_state.sig_figs = 5
if 'history_cache' not in st.session_state: st.session_state.history_cache = ""

# --- 3. CONSTANTS LIBRARY ---
CONSTANTS = {
    "Universal Physics": {
        "c": (299792458, "Speed of Light (m/s)"),
        "g": (9.80665, "Gravity Standard (m/s²)"),
        "G": (6.67430e-11, "Gravitational Const. (N·m²/kg²)"),
        "atm": (101325, "Standard Atmosphere (Pa)"),
    },
    "Thermodynamics (General)": {
        "R_u": (8.314462618, "Univ. Gas Constant (J/mol·K)"),
        "k_B": (1.380649e-23, "Boltzmann Constant (J/K)"),
        "N_A": (6.02214076e23, "Avogadro Number (mol⁻¹)"),
        "sigma": (5.670374419e-8, "Stefan-Boltzmann (W/m²K⁴)"),
        "T_abs": (0, "Absolute Zero (0 K)"),
    },
    "Thermo (Air Properties)": {
        "M_air": (28.97, "Molar Mass Air (g/mol)"),
        "R_air": (287.058, "Gas Const Air (J/kg·K)"),
        "cp_air": (1005, "Specific Heat Cp (J/kg·K)"),
        "cv_air": (718, "Specific Heat Cv (J/kg·K)"),
        "k_air": (1.4, "Specific Heat Ratio (k)"),
    },
    "Thermo (Water/Steam)": {
        "M_h2o": (18.015, "Molar Mass Water (g/mol)"),
        "R_steam": (461.5, "Gas Const Steam (J/kg·K)"),
        "cp_water": (4184, "Specific Heat Liquid (J/kg·K)"),
        "T_crit_h2o": (647.096, "Critical Temp Water (K)"),
        "P_crit_h2o": (22.064e6, "Critical Pressure Water (Pa)"),
    },
    "Particles": {
        "e_c": (1.602176634e-19, "Elementary Charge (C)"),
        "m_e": (9.10938356e-31, "Electron Mass (kg)"),
        "m_p": (1.67262192e-27, "Proton Mass (kg)"),
        "u": (1.660539066e-27, "Atomic Mass Unit (kg)"),
    },
    "Math": {
        "phi": (1.6180339887, "Golden Ratio"),
        "pi": (np.pi, "Pi (3.1415...)"),
        "e": (np.e, "Euler's Number (2.718...)")
    }
}

# --- 4. UNIT CONVERSION LOGIC ---
UNIT_CATEGORIES = {
    "Length": {
        "m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "um": 1e-6,
        "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344
    },
    "Mass": {
        "kg": 1.0, "g": 0.001, "mg": 1e-6, "tonne": 1000.0,
        "lbm": 0.45359237, "slug": 14.5939, "oz": 0.0283495
    },
    "Force": {
        "n": 1.0, "kn": 1000.0, "mn": 1e6,
        "lbf": 4.448222, "kip": 4448.22, "dyn": 1e-5
    },
    "Pressure": {
        "pa": 1.0, "kpa": 1000.0, "mpa": 1e6, "gpa": 1e9,
        "bar": 1e5, "atm": 101325.0, "psi": 6894.757, 
        "torr": 133.322, "mmhg": 133.322
    },
    "Energy": {
        "j": 1.0, "kj": 1000.0, "mj": 1e6,
        "cal": 4.184, "kcal": 4184.0, "btu": 1055.056, 
        "kwh": 3.6e6, "ev": 1.60218e-19
    },
    "Power": {
        "w": 1.0, "kw": 1000.0, "mw": 1e6,
        "hp": 745.7, "hp_met": 735.5
    },
    "Volume": {
        "m3": 1.0, "cm3": 1e-6, "mm3": 1e-9,
        "l": 0.001, "ml": 1e-6, 
        "gal": 0.00378541, "ft3": 0.0283168, "in3": 1.6387e-5
    },
    "Area": {
        "m2": 1.0, "cm2": 1e-4, "mm2": 1e-6, "km2": 1e6, "ha": 10000.0,
        "ft2": 0.092903, "in2": 0.00064516, "acre": 4046.86
    },
    "Speed": {
        "mps": 1.0, "kph": 0.277778, "mph": 0.44704, "kn": 0.514444
    }
}

def perform_conversion(val, u_from, u_to):
    u_from = u_from.lower().replace(" ", "")
    u_to = u_to.lower().replace(" ", "")
    temps = ['c', 'f', 'k', 'r']
    if u_from in temps and u_to in temps:
        if u_from == 'c': k = val + 273.15
        elif u_from == 'f': k = (val - 32) * 5/9 + 273.15
        elif u_from == 'r': k = val * 5/9
        elif u_from == 'k': k = val
        if u_to == 'c': return k - 273.15, "Temperature"
        elif u_to == 'f': return (k - 273.15) * 9/5 + 32, "Temperature"
        elif u_to == 'r': return k * 9/5, "Temperature"
        elif u_to == 'k': return k, "Temperature"
    for category, units in UNIT_CATEGORIES.items():
        if u_from in units and u_to in units:
            return val * units[u_from] / units[u_to], category
    return None, None

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🧮 Victor's Calculator")
    tab_settings, tab_const, tab_help = st.tabs(["⚙️ Set", "⚛️ Const", "📝 Help"])
    
    with tab_settings:
        st.subheader("Display")
        with st.expander("📈 Graph Axis Limits", expanded=False):
            c1, c2 = st.columns(2)
            if 'min_x' not in st.session_state: st.session_state.min_x = -10.0
            st.session_state.min_x = c1.number_input("Min X", value=st.session_state.min_x, step=1.0)
            if 'max_x' not in st.session_state: st.session_state.max_x = 10.0
            st.session_state.max_x = c2.number_input("Max X", value=st.session_state.max_x, step=1.0)
            c3, c4 = st.columns(2)
            if 'min_y' not in st.session_state: st.session_state.min_y = -5.0
            st.session_state.min_y = c3.number_input("Min Y", value=st.session_state.min_y, step=1.0)
            if 'max_y' not in st.session_state: st.session_state.max_y = 5.0
            st.session_state.max_y = c4.number_input("Max Y", value=st.session_state.max_y, step=1.0)

        st.subheader("Calculation")
        trig_mode = st.radio("Angle Unit", ["Radians", "Degrees"], horizontal=True)
        
        c_sf1, c_sf2, c_sf3 = st.columns([1, 2, 1])
        if c_sf1.button("➖", use_container_width=True):
            if st.session_state.sig_figs > 1: st.session_state.sig_figs -= 1
        c_sf2.markdown(f"<div style='text-align:center; padding-top:5px; font-weight:bold; border:1px solid #444; border-radius:5px;'>{st.session_state.sig_figs}</div>", unsafe_allow_html=True)
        if c_sf3.button("➕", use_container_width=True):
            if st.session_state.sig_figs < 20: st.session_state.sig_figs += 1
        
        use_sci = st.checkbox("Scientific Notation", value=False)
        st.divider()
        if 'table_step' not in st.session_state: st.session_state.table_step = 1.0
        st.session_state.table_step = st.number_input("Table Step (Δx)", value=st.session_state.table_step, step=0.1, format="%.3f")
        show_intersect = st.checkbox("🔴 Show Intersections", value=True)
        st.divider()
        # CLOUD MODE: CLEAR RAM
        if st.button("🗑️ Clear History", type="primary", use_container_width=True):
            st.session_state.history_cache = ""
            st.rerun()

    with tab_const:
        st.caption("Constants Library")
        for category, items in CONSTANTS.items():
            with st.expander(category, expanded=False):
                for key, val in items.items():
                    c_name, c_val = st.columns([1, 2])
                    c_name.code(key)
                    c_val.markdown(f"{val[0]:.4g}\n\n<span style='color:gray; font-size:0.8em'>{val[1]}</span>", unsafe_allow_html=True)

    with tab_help:
        st.markdown("### 📖 Command Reference")
        st.info("Commands: `graph`, `solve`, `diff`, `integ`, `deriv`, `calc`, `isolate`, `approx`, `convert`, `lap`, `ilap`")

# --- MEMORY SETUP ---
def np_heaviside(x): return np.heaviside(x, 1)
def np_delta(x): return np.zeros_like(x) 

calc_memory = {"np": np, "math": np, "pi": np.pi, "e": np.e, 
               "sin": np.sin, "cos": np.cos, "tan": np.tan, 
               "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
               "ln": np.log, "step": np_heaviside, "delta": np_delta}

for cat in CONSTANTS.values():
    for key, val in cat.items():
        calc_memory[key] = val[0]

letters = string.ascii_letters
sym_memory = {letter: sp.Symbol(letter) for letter in letters}
sym_memory.update({
    "sp": sp, "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log, "pi": sp.pi,
    "diff": sp.diff, "oo": sp.oo, "e": sp.E, "ln": sp.log,
    "step": sp.Heaviside, "delta": sp.DiracDelta, "erf": sp.erf, "I": sp.I,
    "y0": sp.Symbol("y0"), "yp0": sp.Symbol("yp0"), "Y": sp.Symbol("Y")
})

for cat in CONSTANTS.values():
    for key in cat.keys():
        if key not in sym_memory: sym_memory[key] = sp.Symbol(key)

if trig_mode == "Degrees":
    calc_memory.update({'sin': lambda x: np.sin(np.deg2rad(x)), 'cos': lambda x: np.cos(np.deg2rad(x)), 'tan': lambda x: np.tan(np.deg2rad(x))})
    sym_memory.update({'sin': lambda x: sp.sin(x * sp.pi / 180), 'cos': lambda x: sp.cos(x * sp.pi / 180), 'tan': lambda x: sp.tan(x * sp.pi / 180)})

calc_sym_memory = sym_memory.copy()
calc_sym_memory['f'] = sp.Function('f')
constant_subs = {sym_memory[k]: v[0] for c in CONSTANTS.values() for k,v in c.items() if k in sym_memory}

# --- HELPERS ---
def smart_parse(text):
    try: return parse_expr(text, transformations=(standard_transformations + (implicit_multiplication_application,)), local_dict=sym_memory)
    except: return sp.sympify(text, locals=sym_memory)

def clean_input(text):
    text = text.replace("^", "**")
    text = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', text)
    text = re.sub(r'(\))([a-zA-Z\d\(])', r'\1*\2', text)
    text = re.sub(r'\bas\b', 'a*s', text)
    return text

def translate_diff_eq(text):
    var = 't' if ('t' in text and 'x' not in text) else 'x'
    func_str = f'f({var})'
    def prime_replacer(match): return f"{func_str}.diff({var}, {len(match.group(1))})"
    text = re.sub(r"y('+)", prime_replacer, text)
    text = re.sub(r'\by\b', func_str, text) 
    return text, var

def format_number(val):
    try:
        val_float = float(val)
        if use_sci: return f"{val_float:.{st.session_state.sig_figs}e}"
        else: return f"{val_float:.{st.session_state.sig_figs}g}"
    except: return str(val)

# --- DISPLAY ---
history_container = st.container()

def display_answer(label, exact_val, warning=None):
    st.session_state.ans = exact_val
    with history_container.container(border=True):
        c1, c2 = st.columns([0.05, 0.95])
        c1.markdown("📝")
        with c2:
            st.markdown(f"**{label}**")
            if warning: st.caption(f"⚠️ {warning}")
            try: latex_str = sp.latex(exact_val)
            except: latex_str = str(exact_val)
            st.latex(latex_str)
            if hasattr(exact_val, 'evalf'): st.caption(f"Decimal: {format_number(exact_val)}")

# --- MAIN INPUT (RAM ONLY) ---
user_input = st.text_area("Input Log", height=300, value=st.session_state.history_cache, help="Type commands here")

# If input changes, update RAM and rerun to process
if user_input != st.session_state.history_cache:
    st.session_state.history_cache = user_input
    st.rerun()

# --- PROCESSING LOOP ---
lines = user_input.split('\n')
lines = [l for l in lines if l.strip()]
COLOR_CYCLE = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

for i, line in enumerate(lines):
    line = line.strip()
    raw_content = line 
    cmd = "calc:"
    
    if 'ans' in line.lower() and st.session_state.ans is not None:
        val_to_sub = st.session_state.ans
        if hasattr(val_to_sub, 'rhs'): val_to_sub = val_to_sub.rhs
        line = re.sub(r'\bans\b', f"({val_to_sub})", line, flags=re.IGNORECASE)
        raw_content = line

    if ":" in line:
        parts = line.split(":", 1)
        cmd = parts[0].strip().lower() + ":"
        raw_content = parts[1].strip()
    else:
        first = line.split(" ")[0].lower()
        KNOWN = ["graph", "solve", "diff", "integ", "deriv", "calc", "isolate", "approx", "convert", "lap", "ilap", "partfrac"]
        if first in KNOWN:
            cmd = first + ":"
            raw_content = line[len(first):].strip()
        elif "=" in line: cmd = "solve:"

    try:
        if cmd == "calc:":
            res = smart_parse(clean_input(raw_content))
            if hasattr(res, 'subs'): res = res.subs(constant_subs)
            display_answer(f"Calc: {line}", res)
            
        elif cmd == "solve:":
            l, r = clean_input(raw_content).split("=")
            res = sp.solve(sp.Eq(smart_parse(l), smart_parse(r)))
            display_answer(f"Solve: {line}", res)

        elif cmd == "graph:":
            # GRAPHING LOGIC
            graph_var = 't' if ('t' in raw_content and 'x' not in raw_content) else 'x'
            funcs = raw_content.split(",")
            x_vals = np.linspace(st.session_state.min_x, st.session_state.max_x, 500)
            fig = go.Figure()
            for idx, f in enumerate(funcs):
                clean_f = clean_input(f.strip())
                color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                if "=" in clean_f: # Implicit
                    l, r = clean_f.split("=")
                    expr = smart_parse(l) - smart_parse(r)
                    # Implicit plotting is heavy, skipping for speed in this demo
                    st.caption("Implicit graphing simplified for cloud speed.")
                else:
                    lam_f = sp.lambdify(sym_memory[graph_var], smart_parse(clean_f), modules=["numpy", calc_memory])
                    y_plot = lam_f(x_vals)
                    if isinstance(y_plot, (int, float)): y_plot = np.full_like(x_vals, y_plot)
                    fig.add_trace(go.Scatter(x=x_vals, y=y_plot, name=clean_f, line=dict(color=color)))
            fig.update_layout(height=400, margin=dict(l=20,r=20,t=20,b=20))
            with history_container.container(border=True):
                st.plotly_chart(fig, use_container_width=True)

        elif cmd == "lap:":
            val = smart_parse(clean_input(raw_content))
            res = sp.laplace_transform(val, sym_memory['t'], sym_memory['s'], noconds=True)
            display_answer("Laplace", res)

        elif cmd == "ilap:":
            val = smart_parse(clean_input(raw_content))
            res = sp.inverse_laplace_transform(val, sym_memory['s'], sym_memory['t'], noconds=True)
            display_answer("Inv Laplace", res)

        elif cmd == "diff:":
            clean_ode, var = translate_diff_eq(clean_input(raw_content))
            l, r = clean_ode.split("=")
            res = sp.dsolve(sp.Eq(eval(l, calc_sym_memory), eval(r, calc_sym_memory)))
            display_answer("Diff Eq", res)
            
        elif cmd == "integ:":
            parts = clean_input(raw_content).split(",")
            expr = smart_parse(parts[0])
            var = sym_memory['t'] if ('t' in parts[0] and 'x' not in parts[0]) else sym_memory['x']
            if len(parts) == 3: res = sp.integrate(expr, (var, smart_parse(parts[1]), smart_parse(parts[2])))
            else: res = sp.integrate(expr, var)
            display_answer("Integral", res)

        elif cmd == "deriv:":
            expr = smart_parse(clean_input(raw_content))
            var = sym_memory['t'] if ('t' in raw_content and 'x' not in raw_content) else sym_memory['x']
            display_answer("Derivative", sp.diff(expr, var))

    except Exception as e:
        # Don't show error to keep UI clean, but log it if needed
        # st.error(e) 
        pass

# Scroll hack
st.markdown('<script>window.scrollTo(0,document.body.scrollHeight);</script>', unsafe_allow_html=True)