import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import os
import string
import re
import io
import plotly.graph_objects as go
import plotly.express as px
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from fractions import Fraction
from scipy import stats

# --- OPTIONAL IMPORTS ---
try:
    import docx
except ImportError:
    docx = None

try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
except ImportError:
    ureg = None

# --- 1. PAGE SETUP & PATCHES ---
st.set_page_config(
    layout="wide", 
    page_title="Victor's Calculator",
    page_icon="🧮"
)

# === PATCH: SILENCE THE ANALYTICS WARNING ===
# This tricks the analytics library into using the new command instead of the old one.
if hasattr(st, "query_params"):
    st.experimental_get_query_params = lambda: st.query_params

# --- ANALYTICS SETUP (Safe Import) ---
try:
    import streamlit_analytics
    has_analytics = True
except ImportError:
    has_analytics = False
    class AnalyticsStub:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class StreamlitAnalyticsStub:
        def track(self): return AnalyticsStub()
    streamlit_analytics = StreamlitAnalyticsStub()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* UI Polish */
    .block-container {padding-top: 2rem; padding-bottom: 5rem;}
    h1 {font-size: 2.2rem !important;}
    div[data-testid="stMetricValue"] {font-size: 1.1rem !important;}
    .stAlert {padding: 0.5rem;}
    .stButton button {padding: 0px 10px;} 

    /* --- NAVIGATION MENU (Card Style) --- */
    div[role="radiogroup"] label {
        background-color: #ffffff;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
        display: flex;
    }
    div[role="radiogroup"] label:hover {
        background-color: #fff5f5;
        border-color: #FF4B4B;
        color: #FF4B4B !important;
        transform: translateX(5px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[role="radiogroup"] label:has(div[data-checked="true"]) {
        background-color: #FF4B4B !important;
        border-color: #FF4B4B !important;
        color: white !important;
        transform: translateX(5px);
        box-shadow: 0 2px 5px rgba(255, 75, 75, 0.3);
    }
    div[role="radiogroup"] label:has(div[data-checked="true"]) * {
        color: white !important;
        font-weight: 600;
    }
    div[role="radiogroup"] label > div:first-child {
        display: none;
    }

    /* --- REVERT STYLE FOR "RADIANS/DEGREES" --- */
    div[role="radiogroup"]:has(label:nth-last-child(2):first-child) label {
        background-color: transparent !important;
        padding: 0px !important;
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 0px !important;
        display: inline-flex;
        width: auto;
        transform: none !important;
    }
    div[role="radiogroup"]:has(label:nth-last-child(2):first-child) label:hover {
        background-color: transparent !important;
        border-color: transparent !important;
        color: inherit !important;
        transform: none !important;
        box-shadow: none !important;
    }
    div[role="radiogroup"]:has(label:nth-last-child(2):first-child) label > div:first-child {
        display: flex !important;
    }
    div[role="radiogroup"]:has(label:nth-last-child(2):first-child) label:has(div[data-checked="true"]) {
        background-color: transparent !important;
    }
    div[role="radiogroup"]:has(label:nth-last-child(2):first-child) label:has(div[data-checked="true"]) * {
        color: inherit !important;
    }
    section[data-testid="stSidebar"] div.stExpander div[data-testid="stRadio"] > label {
        display: none !important;
    }

    /* PRINT STYLING */
    @media print {
        .no-print { display: none !important; }
        section[data-testid="stSidebar"] { display: none !important; }
        header { display: none !important; }
        div[data-testid="stChatInput"] { display: none !important; }
        .stAppDeployButton { display: none !important; }
        .main .block-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. STATE MANAGEMENT ---
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

# --- 4. ROBUST UNIT CONVERSION LOGIC ---
PREFIXES = {
    'Y': 1e24, 'Z': 1e21, 'E': 1e18, 'P': 1e15, 'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3, 
    'h': 1e2, 'da': 1e1, 'd': 1e-1, 'c': 1e-2, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 
    'f': 1e-15, 'a': 1e-18, 'z': 1e-21, 'y': 1e-24
}

UNIT_CATEGORIES = {
    "Length": {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "um": 1e-6, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344},
    "Mass": {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "slug": 14.5939, "oz": 0.0283495, "ton": 907.185, "tonne": 1000.0},
    "Force": {"N": 1.0, "lbf": 4.448222, "kip": 4448.22, "dyn": 1e-5},
    "Pressure": {"Pa": 1.0, "bar": 1e5, "atm": 101325.0, "psi": 6894.757, "Torr": 133.322, "mmHg": 133.322},
    "Energy": {"J": 1.0, "cal": 4.184, "Btu": 1055.056, "Wh": 3600.0, "eV": 1.60218e-19},
    "Power": {"W": 1.0, "hp": 745.7},
    "Volume": {"m3": 1.0, "L": 0.001, "gal": 0.00378541, "ft3": 0.0283168, "in3": 1.6387e-5},
    "Area": {"m2": 1.0, "ha": 10000.0, "ft2": 0.092903, "in2": 0.00064516, "acre": 4046.86},
    "Speed": {"m/s": 1.0, "km/h": 0.277778, "mph": 0.44704, "kn": 0.514444},
    "Time": {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0, "y": 31536000.0}
}

def get_simple_factor(u_str):
    u_str = u_str.strip()
    for cat, units in UNIT_CATEGORIES.items():
        if u_str in units: return units[u_str], cat
    if len(u_str) > 1:
        first_char = u_str[0]; two_chars = u_str[:2]
        if two_chars in PREFIXES:
            base = u_str[2:]
            prefix_val = PREFIXES[two_chars]
            for cat, units in UNIT_CATEGORIES.items():
                if base in units: return units[base] * prefix_val, cat
        if first_char in PREFIXES:
            base = u_str[1:]
            prefix_val = PREFIXES[first_char]
            for cat, units in UNIT_CATEGORIES.items():
                if base in units: return units[base] * prefix_val, cat
    return None, None

def perform_conversion(val, u_from_str, u_to_str):
    if ureg:
        try:
            qty_str = f"{val} * {u_from_str}"
            src_qty = ureg.parse_expression(qty_str)
            target_qty = src_qty.to(u_to_str)
            dims = str(target_qty.dimensionality)
            cat = "Temperature" if dims == '[temperature]' else dims
            return target_qty.magnitude, cat
        except Exception: pass

    temps = {'C', 'F', 'K', 'R'}
    if u_from_str in temps and u_to_str in temps:
        k = val
        if u_from_str == 'C': k = val + 273.15
        elif u_from_str == 'F': k = (val - 32) * 5/9 + 273.15
        elif u_from_str == 'R': k = val * 5/9
        if u_to_str == 'C': return k - 273.15, "Temperature"
        elif u_to_str == 'F': return (k - 273.15) * 9/5 + 32, "Temperature"
        elif u_to_str == 'R': return k * 9/5, "Temperature"
        elif u_to_str == 'K': return k, "Temperature"

    if "/" in u_from_str and "/" in u_to_str:
        num_from, den_from = u_from_str.split("/", 1)
        num_to, den_to = u_to_str.split("/", 1)
        f_num_from, c1 = get_simple_factor(num_from)
        f_num_to, c2 = get_simple_factor(num_to)
        f_den_from, c3 = get_simple_factor(den_from)
        f_den_to, c4 = get_simple_factor(den_to)
        if (f_num_from and f_num_to and c1 == c2) and (f_den_from and f_den_to and c3 == c4):
            factor = (f_num_from / f_num_to) / (f_den_from / f_den_to)
            return val * factor, f"{c1}/{c3}"

    factor_from, cat_from = get_simple_factor(u_from_str)
    factor_to, cat_to = get_simple_factor(u_to_str)
    if factor_from is not None and factor_to is not None:
        if cat_from == cat_to:
            return val * factor_from / factor_to, cat_from
    return None, None

def parse_data_input(input_str):
    try:
        clean = re.split(r'[,\s]+', input_str.strip())
        clean = [float(x) for x in clean if x]
        return np.array(clean)
    except: return None

# --- MEMORY ---
def np_heaviside(x): return np.heaviside(x, 1)
def np_delta(x): return np.zeros_like(x) 
def smart_np_log(x, b=10): return np.log(x) / np.log(b)
def format_number(val):
    try:
        val_float = float(val)
        if use_sci: return f"{val_float:.{st.session_state.sig_figs}e}"
        else: return f"{val_float:.{st.session_state.sig_figs}g}"
    except: return str(val)

calc_memory = {"np": np, "math": np, "pi": np.pi, "e": np.e, 
               "sin": np.sin, "cos": np.cos, "tan": np.tan, 
               "sqrt": np.sqrt, "exp": np.exp,
               "ln": np.log, "log": smart_np_log, "log10": np.log10, "log2": np.log2,
               "step": np_heaviside, "delta": np_delta}
for cat in CONSTANTS.values():
    for key, val in cat.items(): calc_memory[key] = val[0]

letters = string.ascii_letters
sym_memory = {letter: sp.Symbol(letter) for letter in letters}
def smart_sp_log(x, b=10): return sp.log(x, b)
sym_memory.update({
    "sp": sp, "exp": sp.exp, "sqrt": sp.sqrt, "pi": sp.pi,
    "diff": sp.diff, "oo": sp.oo, "e": sp.E,
    "ln": sp.log, "log": smart_sp_log, "log10": lambda x: sp.log(x, 10), "log2": lambda x: sp.log(x, 2),
    "step": sp.Heaviside, "delta": sp.DiracDelta, "erf": sp.erf, "I": sp.I,
    "y0": sp.Symbol("y0"), "yp0": sp.Symbol("yp0"), "Y": sp.Symbol("Y"),
    "t": sp.Symbol("t")
})
for cat in CONSTANTS.values():
    for key in cat.keys():
        if key not in sym_memory: sym_memory[key] = sp.Symbol(key)

# --- HELPERS ---
def smart_parse(text):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try: return parse_expr(text, transformations=transformations, local_dict=sym_memory)
    except: return sp.sympify(text, locals=sym_memory)
def clean_input(text):
    text = text.replace("^", "**")
    text = re.sub(r'\blog(\d+)\(([^)]+)\)', r'log(\2, \1)', text)
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
def parse_ivp(ic_list, var_name):
    ics = {}
    f_sym = calc_sym_memory['f']
    var_sym = sym_memory[var_name]
    for ic in ic_list:
        if "=" not in ic: continue
        try:
            match = re.search(r"y('*)\s*\((.*?)\)\s*=\s*(.*)", ic)
            if match:
                primes, point_str, val_str = match.groups()
                point = eval(point_str, {}, calc_memory)
                val = eval(val_str, {}, calc_memory)
                order = len(primes)
                if order == 0: key = f_sym(var_sym).subs(var_sym, point)
                else: key = f_sym(var_sym).diff(var_sym, order).subs(var_sym, point)
                ics[key] = val
        except: continue
    return ics
def display_answer(label, exact_val, warning=None):
    st.session_state.ans = exact_val
    with history_container.container(border=True):
        col_icon, col_content = st.columns([0.05, 0.95])
        with col_icon: st.markdown("📝")
        with col_content:
            st.markdown(f"**{label}**")
            if warning: st.caption(f"⚠️ {warning}")
            try: 
                latex_str = sp.latex(exact_val).replace("\\log", "\\ln") 
                latex_str = latex_str.replace("\\theta", "\\text{step}").replace("\\delta", "\\delta")
            except: latex_str = str(exact_val)
            if hasattr(exact_val, 'is_number') and exact_val.is_number:
                c1, c2 = st.columns([2, 1])
                c1.latex(latex_str)
                c2.metric("Decimal", format_number(exact_val))
            elif isinstance(exact_val, dict):
                items = [f"{sp.latex(k)} = {sp.latex(v).replace('\\log', '\\ln')}" for k, v in exact_val.items()]
                st.latex(", ".join(items))
            elif isinstance(exact_val, list):
                 st.latex(", ".join([sp.latex(r).replace('\\log', '\\ln') for r in exact_val]))
            else: st.latex(latex_str)

# === START TRACKING ===
with streamlit_analytics.track():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🧮 Victor's Calculator")
        
        with st.expander("📱 Navigation", expanded=True):
            app_mode = st.radio(
                "App Mode", 
                ["Chat Calculator", "🧪 Formula Solver", "📈 Graphing", "📊 Statistics"],
                label_visibility="collapsed"
            )
        
        st.divider()

        tab_settings, tab_const, tab_help, tab_log = st.tabs(["⚙️ Set", "⚛️ Const", "📝 Help", "📜 Log"])
        
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
            st.session_state.table_step = st.number_input("Table Step (Δx)", value=st.session_state.table_step, step=0.1, min_value=0.001, format="%.3f")
            show_intersect = st.checkbox("🔴 Show Graph Intersections", value=True)
            st.divider()
            if st.button("🗑️ Clear History", type="primary", use_container_width=True):
                st.session_state.history_cache = ""
                st.rerun()

        # Update Trig Mode Memory
        if trig_mode == "Degrees":
            calc_memory['sin'] = lambda x: np.sin(np.deg2rad(x))
            calc_memory['cos'] = lambda x: np.cos(np.deg2rad(x))
            calc_memory['tan'] = lambda x: np.tan(np.deg2rad(x))
            calc_memory['asin'] = lambda x: np.rad2deg(np.arcsin(x))
            calc_memory['acos'] = lambda x: np.rad2deg(np.arccos(x))
            calc_memory['atan'] = lambda x: np.rad2deg(np.arctan(x))
            sym_memory.update({'sin': lambda x: sp.sin(x*sp.pi/180), 'cos': lambda x: sp.cos(x*sp.pi/180), 'tan': lambda x: sp.tan(x*sp.pi/180)})
        else:
            calc_memory.update({'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan})
            sym_memory.update({'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan})
        
        calc_sym_memory = sym_memory.copy()
        calc_sym_memory['f'] = sp.Function('f')
        constant_subs = {}
        for cat in CONSTANTS.values():
            for key, val in cat.items(): 
                if key in sym_memory and key not in ['pi', 'e']: constant_subs[sym_memory[key]] = val[0]

        with tab_const:
            st.markdown("### 🛠️ Supported Units")
            st.markdown("Prefixes: `n`, `u`, `m`, `k`, `M`, `G`.")
            with st.expander("View Unit Keys (Basic Mode)", expanded=False):
                st.markdown("""
                **Pres:** `Pa, kPa, MPa, psi, atm, bar, mmHg`  
                **Vol:** `m3, ft3, in3, L, mL, gal`  
                **Len:** `m, ft, in, km, mi, yd`  
                **Mass:** `kg, g, lbm, slug, tonne`  
                **Force:** `N, kN, lbf, kip`  
                **Energy:** `J, kJ, Btu, cal, kWh, eV`  
                **Power:** `W, kW, hp`  
                **Temp:** `C, F, K, R`
                """)
            st.divider()
            st.caption("Constants Library")
            for category, items in CONSTANTS.items():
                with st.expander(category, expanded=False):
                    for key, val in items.items():
                        c_name, c_val = st.columns([1, 2])
                        c_name.code(key)
                        c_val.markdown(f"{val[0]:.4g}\n\n<span style='color:gray; font-size:0.8em'>{val[1]}</span>", unsafe_allow_html=True)

        with tab_help:
            st.markdown("### 📖 Comprehensive User Guide")
            with st.expander("🔢 1. Arithmetic & Constants", expanded=True):
                st.markdown("""
                * **Basic:** `+`, `-`, `*`, `/`, `^`
                * **Logs:** `ln(e)` (Base e), `log(100)` (Base 10), `log2(8)` (Base 2)
                * **Constants:** `pi`, `e`, `c`, `g`, `atm`, `R_u`
                """)
            with st.expander("∫ 2. Calculus & ODEs", expanded=False):
                st.markdown("""
                * **Deriv:** `deriv x^3` or `deriv x^2, 2` (at point)
                * **Integ:** `integ x^2` or `integ x^2, 0, 5` (definite)
                * **ODE:** `diff y''+y=0` or `diff y'=y, y(0)=1`
                * **Laplace:** `lap t^2`, `ilap 1/s^2`
                """)
            with st.expander("📏 3. Units & Physics", expanded=False):
                st.markdown("""
                * **Convert:** `100 km/h to m/s`
                * **Prefixes:** `50 MPa to psi`, `100 nm to m`
                * **Compound:** `10 W/s to kW/h` (Auto-splits numerator/denominator)
                """)
            with st.expander("📊 4. Graphing", expanded=False):
                st.markdown("""
                * **Plot:** `graph sin(x), cos(x)`
                * **Implicit:** `graph x^2 + y^2 = 9`
                * **Parametric:** `(sin(t), cos(t))` uses `t` as parameter.
                """)
            with st.expander("🧪 5. Formula & Stats", expanded=False):
                st.markdown("""
                * **Formula Solver:** Switch mode, enter equation `F=m*a`, solve for `m`.
                * **Statistics:** Switch mode, enter data for T-Tests, ANOVA, Regression.
                """)

        with tab_log:
            st.markdown("### 📜 Session History")
            uploaded_log = st.file_uploader("📂 Upload Log", type=["txt", "docx", "doc"], label_visibility="collapsed")
            if uploaded_log and st.button("📥 Load File", use_container_width=True):
                try:
                    content = uploaded_log.getvalue().decode("utf-8")
                    st.session_state.history_cache = content
                    st.rerun()
                except: st.error("Error reading file")
            
            st.text_area("Log", value=st.session_state.history_cache, height=300)
            st.download_button("💾 Download Log", data=st.session_state.history_cache, file_name="history.txt")

    # === APP MODES ===
    if app_mode == "Chat Calculator":
        history_container = st.container()
        if not st.session_state.history_cache:
            with history_container:
                st.markdown("""<div style="text-align: center; color: gray; margin-top: 50px;"><h3>👋 Welcome to Victor's Calculator</h3><p>Start by typing a command below or open the sidebar for help.</p><p style="font-size: 0.9em; color: #666;">⚠️ <strong>Note:</strong> This session is temporary and private. Please save your work using the <strong>Download Log</strong> button in the sidebar.</p></div>""", unsafe_allow_html=True)
        
        lines = st.session_state.history_cache.split('\n')
        lines = [l for l in lines if l.strip()]
        COLOR_CYCLE = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        
        for i, line in enumerate(lines):
            line = line.strip()
            raw_content = line 
            cmd = "calc:"
            if 'ans' in line.lower() and st.session_state.ans is not None:
                val_to_sub = st.session_state.ans
                if hasattr(val_to_sub, 'rhs'): val_to_sub = val_to_sub.rhs
                ans_str = f"({str(val_to_sub)})"
                line = re.sub(r'\bans\b', ans_str, line, flags=re.IGNORECASE)
                raw_content = line
            if ":" in line:
                parts = line.split(":", 1)
                cmd = parts[0].strip().lower() + ":"
                raw_content = parts[1].strip()
            else:
                first_word = line.split(" ")[0].lower()
                KNOWN = ["graph", "solve", "diff", "integ", "deriv", "calc", "isolate", "approx", "convert", "lap", "ilap", "partfrac"]
                if first_word in KNOWN:
                    cmd = first_word + ":"
                    raw_content = line[len(first_word):].strip()
                elif re.match(r"^[\(\)\d\.\-\+eE]+.*to.*", line, re.IGNORECASE):
                     cmd = "convert:"
                     raw_content = line
                elif "y'" in line or re.search(r"y\s*\(", line): cmd = "diff:"
                elif "=" in line: cmd = "solve:"

            try:
                if cmd == "lap:":
                    eq_part_clean = clean_input(raw_content)
                    if "," in eq_part_clean:
                        parts = eq_part_clean.split(",")
                        eq_part = parts[0]; cond_parts = parts[1:]
                    else:
                        eq_part = eq_part_clean; cond_parts = []
                    if "y" in eq_part and ("'" in eq_part or "=" in eq_part):
                        if "=" in eq_part: lhs_str, rhs_str = eq_part.split("=")
                        else: lhs_str, rhs_str = eq_part, "0"
                        def apply_laplace_rules(text):
                            text = text.replace("y''", "(s**2*Y - s*y0 - yp0)")
                            text = text.replace("y'", "(s*Y - y0)")
                            text = re.sub(r'\by\b', 'Y', text)
                            return text
                        def transform_side(side_str):
                            if "y" in side_str or "Y" in side_str:
                                alg_str = apply_laplace_rules(side_str)
                                return smart_parse(alg_str)
                            else:
                                val = smart_parse(side_str)
                                if val == 0: return 0
                                return sp.laplace_transform(val, sym_memory['t'], sym_memory['s'], noconds=True)
                        lhs_expr = transform_side(lhs_str)
                        rhs_expr = transform_side(rhs_str)
                        subs_dict = {}
                        for cond in cond_parts:
                            c_clean = cond.replace("y*(0)", "y0").replace("y(0)", "y0")
                            c_clean = c_clean.replace("y'*(0)", "yp0").replace("y'(0)", "yp0").replace(" ", "")
                            if "=" in c_clean:
                                l_c, r_c = c_clean.split("=")
                                subs_dict[smart_parse(l_c)] = smart_parse(r_c)
                        full_eq = sp.Eq(lhs_expr, rhs_expr)
                        if subs_dict: full_eq = full_eq.subs(subs_dict)
                        try:
                            Y_sol = sp.solve(full_eq, sym_memory['Y'])
                            if Y_sol: display_answer(f"ℒ Laplace Solution Y(s)", Y_sol[0])
                            else: display_answer(f"ℒ Transformed Equation", full_eq)
                        except: display_answer(f"ℒ Transformed Equation", full_eq)
                    else:
                        val = smart_parse(eq_part_clean)
                        res = sp.laplace_transform(val, sym_memory['t'], sym_memory['s'], noconds=True)
                        display_answer(f"ℒ Laplace Transform", res)
                elif cmd == "ilap:":
                    clean_content = clean_input(raw_content)
                    val = smart_parse(clean_content)
                    res = sp.inverse_laplace_transform(val, sym_memory['s'], sym_memory['t'], noconds=True)
                    res = res.replace(sp.Heaviside, lambda *args: 1)
                    display_answer(f"ℒ⁻¹ Inverse Laplace", res)
                elif cmd == "convert:":
                    match = re.search(r"([ \(\)\d\.\-\+eE\*]+)\s*(.+)\s+to\s+(.+)", raw_content, re.IGNORECASE)
                    if match:
                        try:
                            val_str = match.group(1).strip()
                            if val_str.endswith('*'): val_str = val_str[:-1] 
                            val = float(eval(val_str, {}, {}))
                            u_from = match.group(2).strip()
                            u_to = match.group(3).strip()
                            res, cat_name = perform_conversion(val, u_from, u_to)
                            with history_container.container(border=True):
                                c_icon, c_res = st.columns([0.05, 0.95])
                                c_icon.markdown("🔄")
                                if res is not None:
                                    c_res.metric(f"Convert [{cat_name}]", f"{format_number(res)} {u_to}", f"{raw_content}")
                                    st.session_state.ans = res 
                                else: c_res.error(f"Cannot convert '{u_from}' to '{u_to}'. Check case sensitivity (e.g., 'm' vs 'M').")
                        except ValueError:
                            with history_container: st.error("Error parsing number in conversion.")
                    else:
                        with history_container: st.error("Format: `convert [value] [unit] to [unit]`")
                elif cmd == "approx:":
                    clean_content = clean_input(raw_content)
                    val = eval(clean_content, {}, calc_memory)
                    try: res = sp.nsimplify(val, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)], tolerance=0.001, rational=True)
                    except: res = val
                    if isinstance(res, float) or (hasattr(res, 'is_Float') and res.is_Float):
                        frac = Fraction(val).limit_denominator(1000)
                        if frac.denominator < 1000: res = sp.Rational(frac.numerator, frac.denominator)
                    display_answer(f"Approximation: {line}", res)
                elif cmd == "graph:":
                    graph_var = 't' if ('t' in raw_content and 'x' not in raw_content) else 'x'
                    funcs = raw_content.split(",")
                    x_vals = np.linspace(st.session_state.min_x, st.session_state.max_x, 1000)
                    step = st.session_state.table_step
                    x_table = np.arange(st.session_state.min_x, st.session_state.max_x + (step/100), step)
                    fig = go.Figure()
                    df_table = {graph_var: x_table}
                    with history_container.container(border=True):
                        st.subheader(f"📊 Graph: `{line}`")
                        y_arrays_plot = []
                        for idx, func_str in enumerate(funcs):
                            clean_f = clean_input(func_str.strip())
                            line_color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                            if "=" in clean_f and not (clean_f.startswith("y=") or clean_f.endswith("=y")):
                                l, r = clean_f.split("=")
                                expr = smart_parse(l) - smart_parse(r)
                                feature_x = np.linspace(st.session_state.min_x, st.session_state.max_x, 200)
                                feature_y = np.linspace(st.session_state.min_y, st.session_state.max_y, 200)
                                X, Y = np.meshgrid(feature_x, feature_y)
                                f_imp = sp.lambdify((sym_memory['x'], sym_memory['y']), expr, modules=["numpy", calc_memory])
                                Z = f_imp(X, Y)
                                fig.add_trace(go.Contour(z=Z, x=feature_x, y=feature_y, contours=dict(start=0, end=0, size=2, coloring='lines'), line=dict(width=2, color=line_color), showscale=False, name=clean_f))
                            else:
                                if "=" in clean_f: clean_f = clean_f.split("=")[1]
                                try:
                                    sym_expr = smart_parse(clean_f)
                                    lam_f = sp.lambdify(sym_memory[graph_var], sym_expr, modules=["numpy", calc_memory])
                                    y_plot = lam_f(x_vals)
                                    if isinstance(y_plot, (int, float)): y_plot = np.full_like(x_vals, float(y_plot))
                                    elif y_plot.ndim == 0: y_plot = np.full_like(x_vals, float(y_plot))
                                    y_tab = lam_f(x_table)
                                    if isinstance(y_tab, (int, float)): y_tab = np.full_like(x_table, float(y_tab))
                                    elif y_tab.ndim == 0: y_tab = np.full_like(x_table, float(y_tab))
                                    fig.add_trace(go.Scatter(x=x_vals, y=y_plot, mode='lines', name=clean_f, line=dict(color=line_color)))
                                    df_table[clean_f] = y_tab
                                    y_arrays_plot.append(y_plot)
                                except: pass
                        intersect_x, intersect_y = [], []
                        if show_intersect:
                            for j in range(len(y_arrays_plot)):
                                for k in range(j + 1, len(y_arrays_plot)):
                                    y1, y2 = y_arrays_plot[j], y_arrays_plot[k]
                                    diff = y1 - y2
                                    sign_changes = np.where(np.diff(np.signbit(diff)))[0]
                                    for idx in sign_changes:
                                        x0, x1 = x_vals[idx], x_vals[idx+1]
                                        d0, d1 = diff[idx], diff[idx+1]
                                        if (d1 - d0) != 0:
                                            rx = x0 - d0 * (x1 - x0) / (d1 - d0)
                                            ry = np.interp(rx, x_vals, y1)
                                            intersect_x.append(rx); intersect_y.append(ry)
                            if intersect_x:
                                fig.add_trace(go.Scatter(x=intersect_x, y=intersect_y, mode='markers', marker=dict(color='red', size=10, line=dict(width=2, color='black')), name='Intersections'))
                        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", xaxis_title=graph_var, yaxis_title="y", yaxis_range=[st.session_state.min_y, st.session_state.max_y], height=500)
                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")
                        with st.expander(f"📉 Data Points & Intersections ({len(intersect_x)} found)", expanded=False):
                            if intersect_x:
                                 st.markdown("**Intersection Points:**")
                                 st.dataframe(pd.DataFrame({'X': intersect_x, 'Y': intersect_y}), height=150)
                            st.dataframe(pd.DataFrame(df_table))
                elif cmd == "diff:":
                    if "," in raw_content:
                        parts = raw_content.split(",")
                        raw_ode = parts[0]; raw_ics = parts[1:]
                        clean_ode = clean_input(raw_ode)
                        clean_ode, active_var = translate_diff_eq(clean_ode)
                        l, r = clean_ode.split("=")
                        eq = sp.Eq(eval(l, calc_sym_memory), eval(r, calc_sym_memory))
                        f_symbol = calc_sym_memory['f'](sym_memory[active_var])
                        try:
                            ics = parse_ivp(raw_ics, active_var)
                            if not ics: raise Exception("No ICs")
                            res = sp.dsolve(eq, f_symbol, ics=ics)
                            display_answer(f"IVP Solution: {line}", res)
                        except:
                            res = sp.dsolve(eq, f_symbol)
                            display_answer(f"General Solution: {line}", res, warning="Ignored ICs")
                    else:
                        clean_content, active_var = translate_diff_eq(clean_input(raw_content))
                        l, r = clean_content.split("=")
                        indep_sym = sym_memory[active_var]
                        res = sp.dsolve(sp.Eq(eval(l, calc_sym_memory), eval(r, calc_sym_memory)), calc_sym_memory['f'](indep_sym))
                        display_answer(f"General Solution: {line}", res)
                elif cmd == "solve:":
                    clean_content = clean_input(raw_content)
                    if "," in clean_content:
                        eq_list = clean_content.split(",")
                        equations = []
                        all_symbols = set()
                        for p in eq_list:
                            if "=" in p:
                                l, r = p.split("=")
                                lhs, rhs = smart_parse(l), smart_parse(r)
                                eq = sp.Eq(lhs, rhs)
                                equations.append(eq)
                                all_symbols.update(lhs.free_symbols); all_symbols.update(rhs.free_symbols)
                        res = sp.solve(equations, list(all_symbols), dict=True)
                        display_answer(f"System Solution: {line}", res)
                    elif "=" in clean_content:
                        l, r = clean_content.split("=")
                        res = sp.solve(sp.Eq(smart_parse(l), smart_parse(r)))
                        display_answer(f"Roots: {line}", res)
                elif cmd == "calc:":
                    clean_content = clean_input(raw_content)
                    sym_val = smart_parse(clean_content)
                    if hasattr(sym_val, 'subs'):
                         sym_val = sym_val.subs(constant_subs)
                    display_answer(f"Calculation: {line}", sym_val)
                elif cmd == "partfrac:":
                    clean_content = clean_input(raw_content)
                    val = smart_parse(clean_content)
                    res = sp.apart(val)
                    display_answer(f"Partial Fraction Decomposition", res)
                elif cmd == "isolate:":
                    clean_content = clean_input(raw_content)
                    parts = clean_content.split(",")
                    eq_part = parts[0]; target_var = parts[1].strip()
                    l, r = eq_part.split("=")
                    res = sp.solve(sp.Eq(smart_parse(l), smart_parse(r)), smart_parse(target_var))
                    with history_container.container(border=True):
                        st.markdown(f"**Rearrange: `{line}`**")
                        for r in res: st.latex(f"{target_var} = " + sp.latex(r))
                elif cmd == "integ:":
                    clean_content = clean_input(raw_content)
                    parts = clean_content.split(",")
                    expr_str = parts[0].strip()
                    int_var = sym_memory['t'] if ('t' in expr_str and 'x' not in expr_str) else sym_memory['x']
                    sym_expr = smart_parse(expr_str)
                    if len(parts) == 3: 
                        lower = smart_parse(parts[1]); upper = smart_parse(parts[2])
                        res = sp.integrate(sym_expr, (int_var, lower, upper))
                        display_answer(f"Definite Integral: {line}", res)
                    else: 
                        res = sp.integrate(sym_expr, int_var)
                        with history_container.container(border=True):
                            st.markdown(f"**Indefinite Integral: `{line}`**")
                            st.latex(sp.latex(res).replace("\\log", "\\ln").replace("\\theta", "\\text{step}") + " + C")
                elif cmd == "deriv:":
                    clean_content = clean_input(raw_content)
                    if "=" in clean_content:
                        l, r = clean_content.split("=")
                        eq = sp.Eq(smart_parse(l), smart_parse(r))
                        res = sp.idiff(eq.lhs - eq.rhs, sym_memory['y'], sym_memory['x'])
                        display_answer(f"Implicit Derivative (dy/dx)", res)
                    else:
                        parts = clean_content.split(",")
                        expr_str = parts[0].strip()
                        deriv_var = sym_memory['t'] if ('t' in expr_str and 'x' not in expr_str) else sym_memory['x']
                        sym_expr = smart_parse(expr_str)
                        sym_deriv = sp.diff(sym_expr, deriv_var)
                        if len(parts) == 2: 
                            point = smart_parse(parts[1])
                            final_val = sym_deriv.subs(deriv_var, point)
                            display_answer(f"Slope at {deriv_var}={point}", final_val)
                        else:
                            display_answer(f"Derivative Formula: {line}", sym_deriv)

            except Exception as e:
                with history_container:
                    st.error(f"⚠️ Error processing '{line}': {e}")
                    
        new_cmd = st.chat_input("⚡ Type math here (e.g., 'lap y''+y=0', '14.7 psi to kPa')")
        if new_cmd:
            if new_cmd.strip().lower() == "clear": st.session_state.history_cache = ""
            else:
                if st.session_state.history_cache: st.session_state.history_cache += "\n" + new_cmd
                else: st.session_state.history_cache = new_cmd
            st.rerun()

    elif app_mode == "🧪 Formula Solver":
        st.title("🧪 Formula Solver")
        with st.container(border=True):
            raw_formula = st.text_input("Formula (e.g. F = m*a)", key="formula_input")
        if raw_formula:
            try:
                clean_f = clean_input(raw_formula)
                if "=" in clean_f:
                    l, r = clean_f.split("=")
                    lhs = sp.sympify(l)
                    rhs = sp.sympify(r)
                    eq = sp.Eq(lhs, rhs)
                    symbols = eq.free_symbols
                    sorted_syms = sorted(list(symbols), key=lambda s: s.name)
                    sym_map = {s.name: s for s in sorted_syms}
                    sym_names = [s.name for s in sorted_syms]
                    c_sel, c_res = st.columns([1, 2])
                    with c_sel:
                        target_str = st.selectbox("Solve for:", sym_names)
                        target_sym = sym_map[target_str]
                        known_values = {}
                        for s_name in sym_names:
                            if s_name != target_str:
                                val = st.number_input(f"{s_name}", value=0.0, key=f"input_{s_name}")
                                known_values[sym_map[s_name]] = val
                    with c_res:
                        st.markdown("#### 4. Result")
                        try:
                            solutions = sp.solve(eq, target_sym)
                            if not solutions:
                                st.error("Cannot isolate target variable.")
                            else:
                                results_cache = []
                                for i, sol_eq in enumerate(solutions, 1):
                                    st.markdown(f"**Solution {i}:**")
                                    st.latex(f"{target_str} = " + sp.latex(sol_eq))
                                    try:
                                        final_res = sol_eq.subs(known_values).evalf()
                                        res_str = format_number(final_res)
                                        st.info(f"Value: {res_str}")
                                        results_cache.append((sp.latex(sol_eq), res_str))
                                    except Exception as e:
                                        st.warning(f"Could not evaluate numerically: {e}")

                                if st.button("💾 Save Results to Log", use_container_width=True):
                                    input_str = ", ".join([f"{k}={v}" for k, v in known_values.items()])
                                    log_entry = f"Formula: {raw_formula} (Solved for {target_str}) | Inputs: {input_str}"
                                    for idx, (sym_res, num_res) in enumerate(results_cache, 1):
                                        log_entry += f" | Sol {idx}: {num_res}"
                                    if st.session_state.history_cache:
                                        st.session_state.history_cache += "\n" + log_entry
                                    else:
                                        st.session_state.history_cache = log_entry
                                    st.toast("✅ Saved calculations to Session History!")
                        except Exception as e:
                            st.error(f"Calculation Error: {e}")
                else: st.warning("Please enter an equation with an '=' sign.")
            except Exception as e: st.error(f"Error: {e}")

    # === NEW: UPGRADED GRAPHING MODE ===
    elif app_mode == "📈 Graphing":
        st.title("📈 Advanced Graphing")
        
        col_ctrl, col_plot = st.columns([1, 3])
        
        with col_ctrl:
            st.subheader("Functions")
            raw_funcs = st.text_area("Enter equations (one per line):", value="", height=150, placeholder="e.g. sin(x)")
            
            st.subheader("Settings")
            autoscale = st.checkbox("Autoscale Y-Axis", value=True)
            show_grid = st.checkbox("Show Grid", value=True)
            show_intersections = st.checkbox("Show Intersections", value=True)
            
            st.caption("Parametric Range (t): Uses X-Min/Max from Sidebar.")
            
            st.divider()
            st.subheader("Trace")
            trace_val = st.number_input("Evaluate at x =", value=0.0)
            trace_container = st.empty()
            
        with col_plot:
            if raw_funcs:
                funcs = [f.strip() for f in raw_funcs.split('\n') if f.strip()]
                x_vals = np.linspace(st.session_state.min_x, st.session_state.max_x, 1000)
                
                fig = go.Figure()
                y_arrays_plot = []
                valid_funcs = []
                all_y_values = []
                
                df_data = pd.DataFrame({'x': x_vals})
                COLOR_CYCLE = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
                
                for idx, func_str in enumerate(funcs):
                    line_color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                    param_match = re.match(r"\((.+),(.+)\)", func_str)
                    if param_match:
                        try:
                            xt_str, yt_str = param_match.groups()
                            xt_expr = smart_parse(xt_str)
                            yt_expr = smart_parse(yt_str)
                            lam_xt = sp.lambdify(sym_memory['t'], xt_expr, modules=["numpy", calc_memory])
                            lam_yt = sp.lambdify(sym_memory['t'], yt_expr, modules=["numpy", calc_memory])
                            x_p = lam_xt(x_vals)
                            y_p = lam_yt(x_vals)
                            if isinstance(x_p, (int, float)): x_p = np.full_like(x_vals, float(x_p))
                            if isinstance(y_p, (int, float)): y_p = np.full_like(x_vals, float(y_p))
                            fig.add_trace(go.Scatter(x=x_p, y=y_p, mode='lines', name=f"Param: {func_str}", line=dict(color=line_color)))
                            all_y_values.extend(y_p)
                        except: pass
                        continue

                    clean_f = clean_input(func_str)
                    if "=" in clean_f and not (clean_f.startswith("y=") or clean_f.endswith("=y")):
                        try:
                            l, r = clean_f.split("=")
                            expr = smart_parse(l) - smart_parse(r)
                            feature_x = np.linspace(st.session_state.min_x, st.session_state.max_x, 200)
                            feature_y = np.linspace(st.session_state.min_y, st.session_state.max_y, 200)
                            X, Y = np.meshgrid(feature_x, feature_y)
                            f_imp = sp.lambdify((sym_memory['x'], sym_memory['y']), expr, modules=["numpy", calc_memory])
                            Z = f_imp(X, Y)
                            fig.add_trace(go.Contour(z=Z, x=feature_x, y=feature_y, contours=dict(start=0, end=0, size=2, coloring='lines'), line=dict(width=2, color=line_color), showscale=False, name=clean_f))
                        except: pass
                    else:
                        if "=" in clean_f: clean_f = clean_f.split("=")[1]
                        try:
                            sym_expr = smart_parse(clean_f)
                            lam_f = sp.lambdify(sym_memory['x'], sym_expr, modules=["numpy", calc_memory])
                            y_plot = lam_f(x_vals)
                            if isinstance(y_plot, (int, float)): y_plot = np.full_like(x_vals, float(y_plot))
                            fig.add_trace(go.Scatter(x=x_vals, y=y_plot, mode='lines', name=clean_f, line=dict(color=line_color)))
                            y_arrays_plot.append(y_plot)
                            valid_funcs.append((clean_f, lam_f))
                            all_y_values.extend(y_plot)
                            df_data[clean_f] = y_plot
                        except: pass
                
                if show_intersections and len(y_arrays_plot) > 1:
                    intersect_x, intersect_y = [], []
                    for j in range(len(y_arrays_plot)):
                        for k in range(j + 1, len(y_arrays_plot)):
                            y1, y2 = y_arrays_plot[j], y_arrays_plot[k]
                            diff = y1 - y2
                            sign_changes = np.where(np.diff(np.signbit(diff)))[0]
                            for idx in sign_changes:
                                x0, x1 = x_vals[idx], x_vals[idx+1]
                                d0, d1 = diff[idx], diff[idx+1]
                                if (d1 - d0) != 0:
                                    rx = x0 - d0 * (x1 - x0) / (d1 - d0)
                                    ry = np.interp(rx, x_vals, y1)
                                    intersect_x.append(rx); intersect_y.append(ry)
                    if intersect_x:
                        fig.add_trace(go.Scatter(x=intersect_x, y=intersect_y, mode='markers', marker=dict(color='red', size=10, line=dict(width=2, color='black')), name='Intersections'))

                layout_args = dict(xaxis_title="x", yaxis_title="y", height=600, showlegend=True, hovermode="x unified")
                if autoscale and all_y_values:
                    clean_y = [y for y in all_y_values if np.isfinite(y)]
                    if clean_y:
                        y_min, y_max = min(clean_y), max(clean_y)
                        padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                        layout_args['yaxis_range'] = [y_min - padding, y_max + padding]
                else:
                    layout_args['yaxis_range'] = [st.session_state.min_y, st.session_state.max_y]

                fig.update_layout(**layout_args)
                if not show_grid:
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=False)
                    
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📄 Data Table", expanded=False):
                    st.dataframe(df_data, use_container_width=True)
                
                if valid_funcs:
                    with trace_container.container():
                        st.markdown("#### Results")
                        for f_name, f_lam in valid_funcs:
                            try:
                                val = f_lam(trace_val)
                                st.write(f"**{f_name}**: {format_number(val)}")
                            except: pass

    elif app_mode == "📊 Statistics":
        st.title("📊 Statistics Suite")
        stat_tabs = st.tabs(["Descriptive", "Hypothesis Tests", "Regression", "ANOVA / NonParam", "Sampling"])
        
        with stat_tabs[0]:
            st.subheader("Descriptive Statistics")
            d_input = st.text_area("Enter Data (comma separated)", "10, 12, 15, 14, 18, 20, 15, 22")
            data = parse_data_input(d_input)
            if data is not None and len(data) > 0:
                c1, c2 = st.columns([1, 2])
                with c1:
                    stats_res = stats.describe(data)
                    
                    st.markdown("### Summary")
                    st.write(f"**Count:** {stats_res.nobs}")
                    st.write(f"**Mean:** {format_number(stats_res.mean)}")
                    st.write(f"**Median:** {format_number(np.median(data))}")
                    
                    st.write(f"**Variance:** {format_number(stats.tvar(data))}")
                    st.write(f"**Std Dev:** {format_number(np.std(data, ddof=1))}")
                    
                    min_v, max_v = stats_res.minmax
                    st.write(f"**Range:** {format_number(min_v)} - {format_number(max_v)}")
                    
                    q1, q3 = np.percentile(data, [25, 75])
                    iqr = q3 - q1
                    st.divider()
                    st.write(f"**Q1 (25%):** {format_number(q1)}")
                    st.write(f"**Q3 (75%):** {format_number(q3)}")
                    st.write(f"**IQR:** {format_number(iqr)}")
                    
                    st.divider()
                    st.write(f"**Skew:** {format_number(stats_res.skewness)}")
                    st.write(f"**Kurtosis:** {format_number(stats_res.kurtosis)}")
                    
                with c2:
                    fig = px.histogram(data, nbins=10, title="Histogram", labels={'value': 'Data'})
                    fig.add_vline(x=np.mean(data), line_dash="dash", line_color="red", annotation_text="Mean")
                    fig.add_vline(x=np.median(data), line_dash="dash", line_color="green", annotation_text="Median")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig_box = px.box(data, title="Boxplot", points="all")
                    st.plotly_chart(fig_box, use_container_width=True)

        with stat_tabs[1]:
            st.subheader("Hypothesis Testing")
            test_type = st.selectbox("Select Test", ["One-Sample T-Test", "Two-Sample T-Test (Independent)", "Paired T-Test"])
            
            if test_type == "One-Sample T-Test":
                d1_txt = st.text_input("Sample Data", "10, 12, 14, 15, 11")
                mu = st.number_input("Population Mean (μ)", 0.0)
                d1 = parse_data_input(d1_txt)
                if d1 is not None and st.button("Run Test"):
                    res = stats.ttest_1samp(d1, mu)
                    st.write(f"**T-Statistic:** {format_number(res.statistic)}")
                    st.write(f"**P-Value:** {format_number(res.pvalue)}")
                    if res.pvalue < 0.05: st.success("Significant Difference found (Reject H0)")
                    else: st.info("No Significant Difference (Fail to Reject H0)")
            
            elif test_type == "Two-Sample T-Test (Independent)":
                c1, c2 = st.columns(2)
                d1_txt = c1.text_input("Group 1 Data", "10, 12, 14, 15")
                d2_txt = c2.text_input("Group 2 Data", "8, 9, 11, 10")
                d1 = parse_data_input(d1_txt)
                d2 = parse_data_input(d2_txt)
                if d1 is not None and d2 is not None and st.button("Run Test"):
                    res = stats.ttest_ind(d1, d2)
                    st.write(f"**T-Statistic:** {format_number(res.statistic)}")
                    st.write(f"**P-Value:** {format_number(res.pvalue)}")
                    if res.pvalue < 0.05: st.success("Significant Difference found (Reject H0)")
                    else: st.info("No Significant Difference (Fail to Reject H0)")
            
            elif test_type == "Paired T-Test":
                c1, c2 = st.columns(2)
                d1_txt = c1.text_input("Pre-Test", "10, 12, 14")
                d2_txt = c2.text_input("Post-Test", "12, 14, 16")
                d1 = parse_data_input(d1_txt)
                d2 = parse_data_input(d2_txt)
                if d1 is not None and d2 is not None and st.button("Run Test"):
                    if len(d1) == len(d2):
                        res = stats.ttest_rel(d1, d2)
                        st.write(f"**T-Statistic:** {format_number(res.statistic)}")
                        st.write(f"**P-Value:** {format_number(res.pvalue)}")
                        if res.pvalue < 0.05: st.success("Significant Difference found (Reject H0)")
                        else: st.info("No Significant Difference (Fail to Reject H0)")
                    else: st.error("Samples must be equal size.")

        with stat_tabs[2]:
            st.subheader("Regression Analysis")
            reg_type = st.radio("Model Type", ["Linear (y=mx+b)", "Quadratic (y=ax²+bx+c)", "Exponential (y=ae^bx)"])
            
            c1, c2 = st.columns(2)
            x_txt = c1.text_input("X Data", "1, 2, 3, 4, 5")
            y_txt = c2.text_input("Y Data", "2, 4, 9, 16, 25")
            x_dat = parse_data_input(x_txt)
            y_dat = parse_data_input(y_txt)
            
            if x_dat is not None and y_dat is not None and len(x_dat) == len(y_dat):
                if st.button("Calculate Regression"):
                    # Helper for R2
                    def calculate_r2(y_true, y_pred):
                        ss_res = np.sum((y_true - y_pred) ** 2)
                        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                        return 1 - (ss_res / ss_tot)

                    x_plot = np.linspace(min(x_dat), max(x_dat), 100)
                    
                    if "Linear" in reg_type:
                        res = stats.linregress(x_dat, y_dat)
                        y_pred = res.intercept + res.slope * x_dat
                        y_plot = res.intercept + res.slope * x_plot
                        
                        st.markdown(f"**Equation:** $y = {format_number(res.slope)}x + {format_number(res.intercept)}$")
                        st.write(f"**R²:** {format_number(res.rvalue**2)}")
                        st.write(f"**P-Value:** {format_number(res.pvalue)}")
                        
                    elif "Quadratic" in reg_type:
                        coeffs = np.polyfit(x_dat, y_dat, 2)
                        p = np.poly1d(coeffs)
                        y_pred = p(x_dat)
                        y_plot = p(x_plot)
                        r2 = calculate_r2(y_dat, y_pred)
                        
                        st.markdown(f"**Equation:** $y = {format_number(coeffs[0])}x^2 + {format_number(coeffs[1])}x + {format_number(coeffs[2])}$")
                        st.write(f"**R²:** {format_number(r2)}")
                        
                    elif "Exponential" in reg_type:
                        if np.any(y_dat <= 0):
                            st.error("Exponential regression requires all Y values > 0.")
                            y_plot = None
                        else:
                            coeffs = np.polyfit(x_dat, np.log(y_dat), 1)
                            b = coeffs[0]
                            a = np.exp(coeffs[1])
                            y_pred = a * np.exp(b * x_dat)
                            y_plot = a * np.exp(b * x_plot)
                            r2 = calculate_r2(y_dat, y_pred)
                            
                            st.markdown(f"**Equation:** $y = {format_number(a)}e^{{{format_number(b)}x}}$")
                            st.write(f"**R²:** {format_number(r2)}")

                    if 'y_plot' in locals() and y_plot is not None:
                        fig = px.scatter(x=x_dat, y=y_dat, title=f"{reg_type} Fit")
                        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Fit Line', line=dict(color='red')))
                        st.plotly_chart(fig, use_container_width=True)

        with stat_tabs[3]:
            st.subheader("ANOVA / NonParam")
            test_sel = st.selectbox("Test Type", ["One-Way ANOVA", "Mann-Whitney U"])
            if test_sel == "One-Way ANOVA":
                anova_input = st.text_area("Groups (semicolon separated)", "10,12,14; 15,17,19; 20,22,24")
                if st.button("Run ANOVA"):
                    groups = [parse_data_input(g) for g in anova_input.split(";")]
                    groups = [g for g in groups if g is not None and len(g) > 0]
                    if len(groups) >= 2:
                        res = stats.f_oneway(*groups)
                        st.write(f"**F-Stat:** {format_number(res.statistic)}")
                        st.write(f"**P-Value:** {format_number(res.pvalue)}")
                        if res.pvalue < 0.05: st.success("Significant Difference found (Reject H0)")
                        else: st.info("No Significant Difference (Fail to Reject H0)")
            elif test_sel == "Mann-Whitney U":
                c1, c2 = st.columns(2)
                g1 = parse_data_input(c1.text_input("Group A", "1, 2, 3"))
                g2 = parse_data_input(c2.text_input("Group B", "4, 5, 6"))
                if g1 is not None and g2 is not None and st.button("Run Test"):
                    res = stats.mannwhitneyu(g1, g2)
                    st.write(f"**U-Stat:** {res.statistic}")
                    st.write(f"**P-Value:** {format_number(res.pvalue)}")
                    if res.pvalue < 0.05: st.success("Significant Difference found (Reject H0)")
                    else: st.info("No Significant Difference (Fail to Reject H0)")

        with stat_tabs[4]:
            st.subheader("Sampling Demo")
            pop_size = st.slider("Population", 100, 1000, 500)
            sample_size = st.slider("Sample Size", 10, 100, 50)
            df_pop = pd.DataFrame({'ID': range(pop_size), 'Value': np.random.normal(50, 15, pop_size)})
            if st.button("Generate Random Sample"):
                sample = df_pop.sample(sample_size)
                fig = px.histogram(df_pop, x="Value", opacity=0.3)
                fig.add_trace(go.Histogram(x=sample["Value"], marker_color="red", opacity=0.6, name="Sample"))
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("<div class='no-print' style='text-align: center; color: #888; font-size: 0.8em; margin-top: 2rem;'>printer friendly page</div>", unsafe_allow_html=True)
st.markdown('<script>window.scrollTo(0,document.body.scrollHeight);</script>', unsafe_allow_html=True)