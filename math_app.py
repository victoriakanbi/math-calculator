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
    .stButton button {padding: 0px 10px;} /* Compact buttons for Sig Figs */
</style>
""", unsafe_allow_html=True)

# --- 2. STATE MANAGEMENT (RAM ONLY) ---
if 'ans' not in st.session_state: st.session_state.ans = 0
if 'sig_figs' not in st.session_state: st.session_state.sig_figs = 5
# HISTORY IS NOW STORED IN RAM ONLY - PRIVATE PER USER SESSION
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
        
        # --- SIG FIGS UI ---
        st.caption("Significant Figures")
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
            st.session_state.history_cache = "" # CLEAR RAM ONLY
            st.rerun()

    with tab_const:
        st.markdown("### 🛠️ Supported Units")
        st.caption("Auto-detects category (e.g., Mass, Pressure)")
        with st.expander("View Unit Keys", expanded=False):
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
        | **Convert** | Convert units (Implicit supported) | `14.7 psi to kPa` |
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

# --- MAIN LAYOUT ---
history_container = st.container()

# --- INPUT SYSTEM (RAM ONLY) ---
# We now load from st.session_state.history_cache INSTEAD of a file
user_input = st.text_area("Raw Input Log", height=300, value=st.session_state.history_cache, help="Edit this to modify past commands")

# Update cache if user types something
if user_input != st.session_state.history_cache:
    st.session_state.history_cache = user_input
    # NEW: INTERCEPT 'CLEAR' COMMAND
    if user_input.strip().lower().endswith("clear"):
        st.session_state.history_cache = ""
        st.rerun()
    st.rerun()

if st.button("🔄 Rerun Log", use_container_width=True):
    st.rerun()

# --- WELCOME SCREEN ---
if not st.session_state.history_cache:
    with history_container:
        st.markdown("""
        <div style="text-align: center; color: gray; margin-top: 50px;">
            <h3>👋 Welcome to Victor's Calculator</h3>
            <p>Start by typing a command below or open the sidebar for help.</p>
            <p><small>Note: History is temporary and private to this tab.</small></p>
        </div>
        """, unsafe_allow_html=True)

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
        elif re.match(r"^[\(\)\d\.\-\+eE]+\s*[a-zA-Z\d_]+\s+to\s+[a-zA-Z\d_]+", line, re.IGNORECASE):
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

        # --- UNIT CONVERSION ---
        elif cmd == "convert:":
            match = re.search(r"([ \(\)\d\.\-\+eE]+)\s*([a-zA-Z\d_]+)\s+to\s+([a-zA-Z\d_]+)", raw_content, re.IGNORECASE)
            if match:
                try:
                    val = float(eval(match.group(1), {}, {}))
                    u_from = match.group(2)
                    u_to = match.group(3)
                    res, cat_name = perform_conversion(val, u_from, u_to)
                    with history_container.container(border=True):
                        c_icon, c_res = st.columns([0.05, 0.95])
                        c_icon.markdown("🔄")
                        if res is not None:
                            c_res.metric(f"Convert [{cat_name}]", f"{format_number(res)} {u_to}", f"{raw_content}")
                            st.session_state.ans = res 
                        else:
                            c_res.error(f"Cannot convert '{u_from}' to '{u_to}'. Check units in sidebar.")
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
                        fig.add_trace(go.Contour(z=Z, x=feature_x, y=feature_y, contours=dict(start=0, end=0, size=2, coloring='lines'), line=dict(width=
