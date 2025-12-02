import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp
import os
import string
import re
import io
import plotly.graph_objects as go
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from fractions import Fraction

# --- OPTIONAL IMPORTS FOR ROBUST FEATURES ---
# 1. DOCX Support
try:
    import docx
except ImportError:
    docx = None

# 2. PINT Support (For "Convert Literally Anything")
try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    # Optional: Enable contexts if needed, but base is usually enough
except ImportError:
    ureg = None

# 1. PAGE SETUP
st.set_page_config(
    layout="wide", 
    page_title="Victor's Calculator",
    page_icon="🧮"
)

# --- CUSTOM CSS FOR UI POLISH & PRINTING ---
st.markdown("""
<style>
    /* UI Polish */
    .block-container {padding-top: 2rem; padding-bottom: 5rem;}
    h1 {font-size: 2.2rem !important;}
    div[data-testid="stMetricValue"] {font-size: 1.1rem !important;}
    .stAlert {padding: 0.5rem;}
    .stButton button {padding: 0px 10px;} /* Compact buttons for Sig Figs */

    /* PRINT STYLING */
    @media print {
        /* Hide the Sidebar */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        /* Hide the Header (Hamburger menu, etc.) */
        header {
            display: none !important;
        }
        /* Hide the Print Button itself */
        #print-btn-container {
            display: none !important;
        }
        /* Hide the Chat Input at the bottom */
        div[data-testid="stChatInput"] {
            display: none !important;
        }
        /* Hide the deploy button if visible */
        .stAppDeployButton {
            display: none !important;
        }
        
        /* Expand Main Content Area */
        .main .block-container {
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }
    }
</style>

<!-- JS PRINT BUTTON INJECTION -->
<div id="print-btn-container" style="position: fixed; top: 3.5rem; right: 2rem; z-index: 999999;">
    <button onclick="window.print()" style="
        background-color: #ff4b4b; 
        color: white; 
        border: none; 
        padding: 8px 16px; 
        border-radius: 5px; 
        cursor: pointer; 
        font-weight: bold;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.2);
        font-family: sans-serif;
    ">
        🖨️ Print Work
    </button>
</div>
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

# --- 4. UNIT CONVERSION LOGIC (HYBRID) ---
# Hardcoded fallback for common units if Pint is missing
UNIT_CATEGORIES = {
    "Length": {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "um": 1e-6, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344},
    "Mass": {"kg": 1.0, "g": 0.001, "mg": 1e-6, "tonne": 1000.0, "lbm": 0.45359237, "slug": 14.5939, "oz": 0.0283495},
    "Force": {"n": 1.0, "kn": 1000.0, "mn": 1e6, "lbf": 4.448222, "kip": 4448.22, "dyn": 1e-5},
    "Pressure": {"pa": 1.0, "kpa": 1000.0, "mpa": 1e6, "gpa": 1e9, "bar": 1e5, "atm": 101325.0, "psi": 6894.757, "torr": 133.322, "mmhg": 133.322},
    "Energy": {"j": 1.0, "kj": 1000.0, "mj": 1e6, "cal": 4.184, "kcal": 4184.0, "btu": 1055.056, "kwh": 3.6e6, "ev": 1.60218e-19},
    "Power": {"w": 1.0, "kw": 1000.0, "mw": 1e6, "hp": 745.7, "hp_met": 735.5},
    "Volume": {"m3": 1.0, "cm3": 1e-6, "mm3": 1e-9, "l": 0.001, "ml": 1e-6, "gal": 0.00378541, "ft3": 0.0283168, "in3": 1.6387e-5},
    "Area": {"m2": 1.0, "cm2": 1e-4, "mm2": 1e-6, "km2": 1e6, "ha": 10000.0, "ft2": 0.092903, "in2": 0.00064516, "acre": 4046.86},
    "Speed": {"mps": 1.0, "m/s": 1.0, "kph": 0.277778, "km/h": 0.277778, "mph": 0.44704, "kn": 0.514444},
}

def perform_conversion(val, u_from_str, u_to_str):
    # 1. ROBUST METHOD: Use PINT if installed (Handles compound units like kg*m/s^2)
    if ureg:
        try:
            # Clean up inputs for Pint (Pint prefers '**' but handles '^' usually)
            # We treat 'C', 'F' specially in Pint usually, but standard parser handles degC/degF
            qty_str = f"{val} * {u_from_str}"
            src_qty = ureg.parse_expression(qty_str)
            target_qty = src_qty.to(u_to_str)
            
            # Formatting
            dims = str(target_qty.dimensionality)
            if dims == '[temperature]': cat = "Temperature"
            elif dims == '[length]': cat = "Length"
            elif dims == '[mass]': cat = "Mass"
            elif dims == '[time]': cat = "Time"
            else: cat = dims # e.g., "[length] / [time] ** 2"
            
            return target_qty.magnitude, cat
        except Exception:
            # Fall through to manual method if Pint fails or doesn't recognize unit
            pass

    # 2. FALLBACK METHOD: Hardcoded Dictionary
    u_from = u_from_str.lower().replace(" ", "")
    u_to = u_to_str.lower().replace(" ", "")
    
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
            base_val = val * units[u_from]
            final_val = base_val / units[u_to]
            return final_val, category
            
    return None, None

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🧮 Victor's Calculator")
    
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
            if 'log_area_widget' in st.session_state:
                del st.session_state.log_area_widget
            st.rerun()

    with tab_const:
        st.markdown("### 🛠️ Supported Units")
        if ureg:
            st.success("✅ Conversions are case sensitive")
            st.caption("You can convert almost any physics unit (e.g., `kg*m/s^2` to `N`, `furlong/fortnight` to `m/s`).")
        else:
            st.warning("⚠️ Basic Mode (Install `pint` for robust units)")
            st.caption("Only basic units supported. Run `pip install Pint` to unlock everything.")
            
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
        st.markdown("### 📖 Command Reference")
        st.info("Most commands work without the keyword (e.g., just type `x^2+y^2=9`).")
        st.markdown("""
        | Command | Description & Syntax | Example |
        | :--- | :--- | :--- |
        | **Clear** | Wipes history immediately | `clear` |
        | **Convert** | Convert units (Implicit supported) | `1 m/s^2 to km/h^2` |
        | **Temp** | Temperature conversion | `100 F to C` |
        | **Laplace** | Laplace ($t \\to s$) or ODE | `lap t^2` or `lap y''+y=0, y(0)=1` |
        | **Inv Lap** | Inverse Laplace ($s \\to t$) | `ilap 1/s^2` |
        | **PartFrac**| Partial Fraction Decomposition | `partfrac 1/(s^2+s)` |
        | **Calc** | Basic Math & Constants | `calc 10 * g + P_atm` |
        | **Approx** | Decimal approximation | `approx pi + sqrt(2)` |
        | **Graph** | Plot functions (Explicit/Implicit) | `graph sin(x), x^2+y^2=9` |
        | **Solve** | Find Roots or Solve Systems | `solve x^2-4=0`<br>`solve x+y=5, x-y=1` |
        | **Diff Eq** | Solve ODEs (General or IVP) | `diff y''+y=0`<br>`diff y'=y, y(0)=1` |
        | **Integ** | Integral (Definite/Indefinite) | `integ x^2` (Indef)<br>`integ x^2, 0, 5` (Def) |
        | **Deriv** | Derivative (Formula/Point/Imp) | `deriv x^3` (Formula)<br>`deriv x^2, 2` (Point)<br>`deriv x^2+y^2=1` (Implicit) |
        | **Isolate** | Rearrange physics formulas | `isolate F=G*m1*m2/r^2, m1` |
        """, unsafe_allow_html=True)

    # --- RESTORED LOG TAB (WITH UPLOAD) ---
    with tab_log:
        st.markdown("### 📜 Session History")
        
        # --- NEW: FILE UPLOADER ---
        uploaded_log = st.file_uploader("📂 Upload Log", type=["txt", "docx", "doc"], label_visibility="collapsed")
        
        # Explanatory Note
        st.info("ℹ️ **Load History:** Upload a file or paste your commands below to restore your session.")

        if uploaded_log and st.button("📥 Load File into Log", use_container_width=True):
            content = ""
            try:
                # Handle .docx
                if uploaded_log.name.endswith(".docx"):
                    if docx:
                        doc = docx.Document(uploaded_log)
                        content = "\n".join([para.text for para in doc.paragraphs])
                    else:
                        st.error("Missing `python-docx` library. Cannot read .docx files.")
                # Handle .txt or text-based .doc
                else:
                    # Try utf-8, fallback to latin-1
                    try:
                        content = uploaded_log.getvalue().decode("utf-8")
                    except:
                        content = uploaded_log.getvalue().decode("latin-1")
                
                if content:
                    st.session_state.history_cache = content
                    # Force sync next render
                    st.session_state.log_area_widget = content
                    st.success("Log loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        st.divider()

        # Helper to sync text area with cache AND handle specific commands like 'clear'
        def update_cache_from_area():
            new_val = st.session_state.log_area_widget
            if new_val.strip().lower().endswith("clear"):
                st.session_state.history_cache = ""
                st.session_state.log_area_widget = "" 
            else:
                st.session_state.history_cache = new_val

        # SYNC WIDGET WITH CACHE BEFORE RENDER
        if 'history_cache' in st.session_state:
             st.session_state.log_area_widget = st.session_state.history_cache

        # 1. The editable log
        st.text_area(
            "Raw Input Log", 
            value=st.session_state.history_cache, 
            height=300, 
            key="log_area_widget",
            on_change=update_cache_from_area,
            help="Edit this to modify past commands. Type 'clear' to wipe."
        )
        
        if st.button("🔄 Rerun Log", use_container_width=True):
            st.rerun()
            
        st.divider()
        
        # 2. DOWNLOAD BUTTON
        st.download_button(
            label="💾 Download Log as .txt",
            data=st.session_state.history_cache,
            file_name="calculator_history.txt",
            mime="text/plain",
            use_container_width=True
        )

# --- MEMORY SETUP ---
def np_heaviside(x): return np.heaviside(x, 1)
def np_delta(x): return np.zeros_like(x) 

# Base Functions (Numeric)
calc_memory = {"np": np, "math": np, "pi": np.pi, "e": np.e, 
               "sin": np.sin, "cos": np.cos, "tan": np.tan, 
               "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
               "ln": np.log, "step": np_heaviside, "delta": np_delta}

# INJECT CONSTANTS INTO NUMERIC MEMORY
for cat in CONSTANTS.values():
    for key, val in cat.items():
        calc_memory[key] = val[0]

# SYMBOLIC MEMORY
letters = string.ascii_letters
sym_memory = {letter: sp.Symbol(letter) for letter in letters}
sym_memory.update({
    "sp": sp, "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log, "pi": sp.pi,
    "diff": sp.diff, "oo": sp.oo, "e": sp.E, "ln": sp.log,
    "step": sp.Heaviside, "delta": sp.DiracDelta, "erf": sp.erf, "I": sp.I,
    "y0": sp.Symbol("y0"), "yp0": sp.Symbol("yp0"), 
    "Y": sp.Symbol("Y")
})

# INJECT CONSTANT NAMES INTO SYMBOLIC MEMORY
for cat in CONSTANTS.values():
    for key in cat.keys():
        if key not in sym_memory:
            sym_memory[key] = sp.Symbol(key)

# TRIG UNIT LOGIC
if trig_mode == "Degrees":
    calc_memory['sin'] = lambda x: np.sin(np.deg2rad(x))
    calc_memory['cos'] = lambda x: np.cos(np.deg2rad(x))
    calc_memory['tan'] = lambda x: np.tan(np.deg2rad(x))
    calc_memory['asin'] = lambda x: np.rad2deg(np.arcsin(x))
    calc_memory['acos'] = lambda x: np.rad2deg(np.arccos(x))
    calc_memory['atan'] = lambda x: np.rad2deg(np.arctan(x))
    sym_memory['sin'] = lambda x: sp.sin(x * sp.pi / 180)
    sym_memory['cos'] = lambda x: sp.cos(x * sp.pi / 180)
    sym_memory['tan'] = lambda x: sp.tan(x * sp.pi / 180)
    sym_memory['asin'] = lambda x: 180/sp.pi * sp.asin(x)
    sym_memory['acos'] = lambda x: 180/sp.pi * sp.acos(x)
    sym_memory['atan'] = lambda x: 180/sp.pi * sp.atan(x)
else:
    calc_memory.update({'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan})
    sym_memory.update({'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan})

calc_sym_memory = sym_memory.copy()
calc_sym_memory['f'] = sp.Function('f')

# BRIDGE FOR CALCULATOR MODE
constant_subs = {}
for cat in CONSTANTS.values():
    for key, val in cat.items():
        if key in sym_memory and key not in ['pi', 'e']:
            constant_subs[sym_memory[key]] = val[0]

# --- HELPERS ---
def smart_parse(text):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try: return parse_expr(text, transformations=transformations, local_dict=sym_memory)
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

def format_number(val):
    try:
        val_float = float(val)
        if use_sci: return f"{val_float:.{st.session_state.sig_figs}e}"
        else: return f"{val_float:.{st.session_state.sig_figs}g}"
    except: return str(val)

# --- UI HELPER: CARD DISPLAY ---
def display_answer(label, exact_val, warning=None):
    # SAVE RESULT TO 'ANS'
    st.session_state.ans = exact_val
    
    with history_container.container(border=True):
        col_icon, col_content = st.columns([0.05, 0.95])
        with col_icon:
            st.markdown("📝") # Icon
        with col_content:
            st.markdown(f"**{label}**")
            if warning: st.caption(f"⚠️ {warning}")
            
            try: latex_str = sp.latex(exact_val).replace("\\log", "\\ln").replace("\\theta", "\\text{step}").replace("\\delta", "\\delta")
            except: latex_str = str(exact_val)
            
            if hasattr(exact_val, 'is_number') and exact_val.is_number:
                c1, c2 = st.columns([2, 1])
                c1.latex(latex_str)
                c2.metric("Decimal", format_number(exact_val))
            
            elif isinstance(exact_val, dict):
                items = [f"{sp.latex(k)} = {sp.latex(v).replace('\\log', '\\ln')}" for k, v in exact_val.items()]
                st.latex(", ".join(items))
                vals = []
                for k,v in exact_val.items():
                    if hasattr(v, 'is_number') and v.is_number:
                        vals.append(f"{k} ≈ {format_number(v)}")
                if vals: st.caption(" | ".join(vals))
            
            elif isinstance(exact_val, list):
                 if exact_val and isinstance(exact_val[0], dict):
                     for i, s in enumerate(exact_val):
                         items = [f"{sp.latex(k)}={sp.latex(v)}" for k,v in s.items()]
                         st.latex(f"Set {i+1}: " + ", ".join(items))
                 else:
                     st.latex(", ".join([sp.latex(r).replace('\\log', '\\ln') for r in exact_val]))
                     vals = [format_number(v) for v in exact_val if hasattr(v, 'is_number') and v.is_number]
                     if vals: st.caption("Decimals: " + ", ".join(vals))
            else: 
                st.latex(latex_str)

# --- MAIN LAYOUT ---
history_container = st.container()

# --- WELCOME SCREEN ---
if not st.session_state.history_cache:
    with history_container:
        st.markdown("""
        <div style="text-align: center; color: gray; margin-top: 50px;">
            <h3>👋 Welcome to Victor's Calculator</h3>
            <p>Start by typing a command below or open the sidebar for help.</p>
            <p style="font-size: 0.9em; color: #666;">⚠️ <strong>Note:</strong> This session is temporary and private. Please save your work using the <strong>Download Log</strong> button in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)

# --- MAIN LOGIC LOOP ---
lines = st.session_state.history_cache.split('\n')
lines = [l for l in lines if l.strip()]

COLOR_CYCLE = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

for i, line in enumerate(lines):
    line = line.strip()
    raw_content = line 
    cmd = "calc:"
    
    # --- ANS SUBSTITUTION LOGIC (IMPROVED) ---
    if 'ans' in line.lower() and st.session_state.ans is not None:
        val_to_sub = st.session_state.ans
        # If 'ans' is an Equation (e.g. f(t) = ...), extract just the RHS for graphing/calc
        if hasattr(val_to_sub, 'rhs'):
            val_to_sub = val_to_sub.rhs
        
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
        # --- ROBUST CONVERSION PARSING (UNIT1 TO UNIT2) ---
        elif re.match(r"^[\(\)\d\.\-\+eE]+.*to.*", line, re.IGNORECASE):
             cmd = "convert:"
             raw_content = line
        elif "y'" in line or re.search(r"y\s*\(", line): cmd = "diff:"
        elif "=" in line: cmd = "solve:"

    try:
        # --- LAPLACE TRANSFORM ---
        if cmd == "lap:":
            eq_part_clean = clean_input(raw_content)
            if "," in eq_part_clean:
                parts = eq_part_clean.split(",")
                eq_part = parts[0]
                cond_parts = parts[1:]
            else:
                eq_part = eq_part_clean
                cond_parts = []

            if "y" in eq_part and ("'" in eq_part or "=" in eq_part):
                if "=" in eq_part:
                    lhs_str, rhs_str = eq_part.split("=")
                else:
                    lhs_str, rhs_str = eq_part, "0"

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
                if subs_dict:
                    full_eq = full_eq.subs(subs_dict)
                
                try:
                    Y_sol = sp.solve(full_eq, sym_memory['Y'])
                    if Y_sol:
                        display_answer(f"ℒ Laplace Solution Y(s)", Y_sol[0])
                    else:
                        display_answer(f"ℒ Transformed Equation", full_eq)
                except:
                    display_answer(f"ℒ Transformed Equation", full_eq)
            
            else:
                val = smart_parse(eq_part_clean)
                t_sym = sym_memory['t']
                s_sym = sym_memory['s']
                res = sp.laplace_transform(val, t_sym, s_sym, noconds=True)
                display_answer(f"ℒ Laplace Transform", res)

        # --- PARTIAL FRACTION ---
        elif cmd == "partfrac:":
            clean_content = clean_input(raw_content)
            val = smart_parse(clean_content)
            res = sp.apart(val)
            display_answer(f"Partial Fraction Decomposition", res)

        # --- INVERSE LAPLACE ---
        elif cmd == "ilap:":
            clean_content = clean_input(raw_content)
            val = smart_parse(clean_content)
            t_sym = sym_memory['t']
            s_sym = sym_memory['s']
            res = sp.inverse_laplace_transform(val, s_sym, t_sym, noconds=True)
            res = res.replace(sp.Heaviside, lambda *args: 1)
            display_answer(f"ℒ⁻¹ Inverse Laplace", res)

        # --- UNIT CONVERSION (ROBUST) ---
        elif cmd == "convert:":
            # regex for: (number) (unit1) to (unit2)
            match = re.search(r"([ \(\)\d\.\-\+eE\*]+)\s*(.+)\s+to\s+(.+)", raw_content, re.IGNORECASE)
            if match:
                try:
                    val_str = match.group(1).strip()
                    if val_str.endswith('*'): val_str = val_str[:-1] # cleanup
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
                        else:
                            c_res.error(f"Cannot convert '{u_from}' to '{u_to}'. Check the case, it is case sensitive.")
                except ValueError:
                    with history_container: st.error("Error parsing number in conversion.")
            else:
                with history_container: st.error("Format: `convert [value] [unit] to [unit]`")

        # --- APPROX ---
        elif cmd == "approx:":
            clean_content = clean_input(raw_content)
            val = eval(clean_content, {}, calc_memory)
            try:
                res = sp.nsimplify(val, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)], tolerance=0.001, rational=True)
            except: res = val
            if isinstance(res, float) or (hasattr(res, 'is_Float') and res.is_Float):
                frac = Fraction(val).limit_denominator(1000)
                if frac.denominator < 1000: res = sp.Rational(frac.numerator, frac.denominator)
            display_answer(f"Approximation: {line}", res)

        # --- GRAPHING ---
        elif cmd == "graph:":
            # Explicitly check if variable is t or x
            graph_var = 't' if ('t' in raw_content and 'x' not in raw_content) else 'x'
            funcs = raw_content.split(",")
            x_vals = np.linspace(st.session_state.min_x, st.session_state.max_x, 1000)
            step = st.session_state.table_step
            x_table = np.arange(st.session_state.min_x, st.session_state.max_x + (step/100), step)
            graph_mem = calc_memory.copy()
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

        # --- DIFF EQ ---
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

        # --- SOLVE ---
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

        # --- CALC ---
        elif cmd == "calc:":
            clean_content = clean_input(raw_content)
            sym_val = smart_parse(clean_content)
            if hasattr(sym_val, 'subs'):
                 sym_val = sym_val.subs(constant_subs)
            display_answer(f"Calculation: {line}", sym_val)

        # --- ISOLATE ---
        elif cmd == "isolate:":
            clean_content = clean_input(raw_content)
            parts = clean_content.split(",")
            eq_part = parts[0]; target_var = parts[1].strip()
            l, r = eq_part.split("=")
            res = sp.solve(sp.Eq(smart_parse(l), smart_parse(r)), smart_parse(target_var))
            with history_container.container(border=True):
                st.markdown(f"**Rearrange: `{line}`**")
                for r in res: st.latex(f"{target_var} = " + sp.latex(r))

        # --- INTEG ---
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

        # --- DERIV ---
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

# --- INPUT BAR ---
new_cmd = st.chat_input("⚡ Type math here (e.g., 'lap y''+y=0', '14.7 psi to kPa')")
if new_cmd:
    if new_cmd.strip().lower() == "clear":
        st.session_state.history_cache = ""
        # Don't touch log_area_widget here, it syncs at start of next run
    else:
        if st.session_state.history_cache:
            st.session_state.history_cache += "\n" + new_cmd
        else:
            st.session_state.history_cache = new_cmd
            
    st.rerun()

# Scroll to bottom
st.markdown('<script>window.scrollTo(0,document.body.scrollHeight);</script>', unsafe_allow_html=True)