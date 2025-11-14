#!/usr/bin/env python3
"""
PropPulse+ v2025.7 — NBA Prop Analyzer (Neon Edition)
Mobile-Optimized | Blue–Red Neon Theme | Center Tabs UI
"""

import os
import io
import base64
from datetime import datetime, timedelta
from contextlib import redirect_stdout

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ===============================
# 🔗 Google Sheets config (Live EV Board)
# ===============================
SHEET_ID = "1SHuoEg331k_dcrgBoc7y8gWbgw1QTKHFJRzzNRqiOnE"
SHEET_GID = "1954146299"  # your main EV sheet tab
SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
    f"?format=csv&gid={SHEET_GID}"
)

# ===============================
# 🧠 Import model
# ===============================
try:
    import prop_ev as pe
except ImportError as e:
    st.error(f"❌ Failed to import prop_ev.py: {e}")
    st.stop()

# ===============================
# ⚙️ PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="PropPulse+ | NBA Props",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# 🎨 GLOBAL STYLING (Neon Blue–Red)
# ===============================
def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #020617;
            --bg-surface: #020617;
            --bg-surface-alt: #020617;
            --accent-blue: #3b82f6;
            --accent-red: #ef4444;
            --accent-purple: #6366f1;
            --text-primary: #f9fafb;
            --text-muted: #9ca3af;
            --border-subtle: #1f2937;
        }

        .main {
            background: radial-gradient(circle at top, #0b1120 0, #020617 55%, #000000 100%);
            color: var(--text-primary);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
            max-width: 1200px;
        }

        section[data-testid="stSidebar"] {
            background: #020617;
            border-right: 1px solid var(--border-subtle);
        }

        section[data-testid="stSidebar"] .stButton button,
        section[data-testid="stSidebar"] .stSelectbox select,
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input {
            border-radius: 10px;
        }

        h1, h2, h3, h4 {
            color: var(--text-primary);
            letter-spacing: 0.03em;
        }

        .pulse-gradient {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.35) 0, transparent 45%),
                radial-gradient(circle at top right, rgba(239,68,68,0.35) 0, transparent 45%);
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.35);
            padding: 1.1rem 1.3rem;
        }

        .metric-card {
            background: rgba(15,23,42,0.96);
            border-radius: 16px;
            padding: 0.85rem 1.0rem;
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 18px 50px rgba(15,23,42,0.85);
        }

        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--text-muted);
        }

        .metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .metric-sub {
            font-size: 0.88rem;
            color: var(--text-muted);
        }

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox select {
            background: rgba(15,23,42,0.96) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            padding: 10px 12px !important;
            font-size: 0.95rem !important;
        }

        .stButton button {
            border-radius: 999px !important;
            padding: 0.6rem 1.1rem !important;
            border: 1px solid rgba(148,163,184,0.4) !important;
        }

        .primary-btn button {
            background: linear-gradient(135deg, #3b82f6, #ef4444) !important;
            border: none !important;
            font-weight: 600 !important;
        }

        .stDataFrame, .stTable {
            border-radius: 14px;
            overflow: hidden;
        }

        .footer {
            margin-top: 2rem;
            padding-top: 1.25rem;
            border-top: 1px solid rgba(148,163,184,0.35);
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }

            h1 { font-size: 1.35rem; }
            h2 { font-size: 1.1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# ===============================
# 🖼️ LOGO HANDLING
# ===============================
def get_logo_base64():
    logo_path = "proppulse_logo.png"
    if not os.path.exists(logo_path):
        return None
    with open(logo_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def render_header():
    logo_b64 = get_logo_base64()

    with st.container():
        cols = st.columns([1, 4])
        with cols[0]:
            if logo_b64:
                st.markdown(
                    f"""
                    <img src="data:image/png;base64,{logo_b64}"
                         style="width:70px;height:auto;border-radius:14px;
                                border:1px solid rgba(148,163,184,0.45);" />
                    """,
                    unsafe_allow_html=True,
                )
        with cols[1]:
            st.markdown(
                """
                <div class="pulse-gradient">
                    <div style="
                        font-size:0.78rem;
                        text-transform:uppercase;
                        color:#9ca3af;
                        letter-spacing:0.16em;
                        margin-bottom:0.25rem;">
                        PropPulse+ · NBA Player Prop Engine
                    </div>
                    <div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:0.45rem;">
                        <span style="font-size:1.32rem;font-weight:720;">
                            Data-Calibrated Player Prop Analyzer
                        </span>
                        <span style="font-size:0.85rem;color:#9ca3af;">
                            Form & matchup–aware · Market-calibrated · EV-driven
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ===============================
# 🧩 HELPERS FOR MODEL CALLS
# ===============================
def safe_load_settings():
    if hasattr(pe, "load_settings"):
        try:
            return pe.load_settings()
        except Exception as e:
            st.warning(f"⚠️ Could not load settings from prop_ev.py: {e}")
            return {}
    return {}


def run_single_prop(player, stat, line, odds, debug_mode=True):
    settings = safe_load_settings()
    try:
        if hasattr(pe, "analyze_single_prop"):
            return pe.analyze_single_prop(
                player=player,
                stat=stat,
                line=line,
                odds=odds,
                settings=settings,
                debug_mode=debug_mode,
            )
        buf = io.StringIO()
        with redirect_stdout(buf):
            if hasattr(pe, "main"):
                pe.main()
        return {"raw_output": buf.getvalue()}
    except Exception as e:
        st.error(f"❌ Error while running model: {e}")
        return None


def run_batch_from_df(df_input, debug_mode=True):
    settings = safe_load_settings()
    try:
        if hasattr(pe, "analyze_batch_df"):
            return pe.analyze_batch_df(df_input, settings=settings, debug_mode=debug_mode)
        elif hasattr(pe, "analyze_batch"):
            return pe.analyze_batch(df_input, settings=settings, debug_mode=debug_mode)
        else:
            st.warning("⚠️ Batch function not found in prop_ev.py. Expected analyze_batch_df or analyze_batch.")
            return None
    except Exception as e:
        st.error(f"❌ Error while running batch analysis: {e}")
        return None

# ===============================
# 📊 SINGLE PROP UI
# ===============================
def single_prop_view():
    render_header()
    st.markdown("### 🎯 Single Prop Analyzer")

    with st.container():
        c1, c2 = st.columns([1.3, 1])

        with c1:
            player = st.text_input("Player name", placeholder="e.g., Cade Cunningham")
            stat = st.selectbox(
                "Stat type",
                ["PTS", "REB", "AST", "PRA", "REB+AST", "PTS+REB", "PTS+AST", "FG3M"],
                index=0,
            )

            left, right = st.columns(2)
            with left:
                line = st.number_input("Line", min_value=0.0, step=0.5, format="%.1f")
            with right:
                odds = st.text_input("Odds (US)", value="-110", help="Enter like -110 or +100")

            debug_mode = st.checkbox("Enable debug mode", value=True)

        with c2:
            st.markdown("##### Quick notes")
            st.write(
                "Uses recent form, season context, and matchup data to flag soft vs sharp lines. "
                "Probabilities are calibrated to realistic NBA distributions instead of raw hit rates."
            )
            st.caption("Tip: Alternate lines are fine — just adjust the line number and keep the same odds.")

    st.markdown("---")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run = st.button("🚀 Analyze Prop", type="primary", use_container_width=True)

    if not run:
        return

    if not player.strip():
        st.error("Please enter a valid player name.")
        return

    with st.spinner("Running PropPulse+ model…"):
        result = run_single_prop(player, stat, line, odds, debug_mode=debug_mode)

    if result is None:
        return

    if "raw_output" in result:
        st.code(result["raw_output"], language="text")
        return

    # Normalize into DataFrame
    if isinstance(result, dict):
        df_res = pd.DataFrame([result])
    elif isinstance(result, pd.DataFrame):
        df_res = result.copy()
    else:
        st.write(result)
        return

    # Extract key values (robust to different column names)
    proj = df_res.get("projection", df_res.get("Projection", pd.Series([None]))).iloc[0]
    direction = df_res.get("direction", df_res.get("Direction", pd.Series([""]))).iloc[0]
    ev_cents = df_res.get("EV¢", df_res.get("EV", pd.Series([None]))).iloc[0]
    model_prob = df_res.get("p_model", df_res.get("Model Prob", df_res.get("Model_Prob", pd.Series([None])))).iloc[0]
    book_prob = df_res.get("p_book", df_res.get("Book Prob", df_res.get("Book_Prob", pd.Series([None])))).iloc[0]
    confidence = df_res.get("confidence", df_res.get("Confidence", pd.Series([None]))).iloc[0]
    opponent = df_res.get("opponent", df_res.get("Opponent", pd.Series(["–"]))).iloc[0]
    position = df_res.get("position", df_res.get("Position", pd.Series(["–"]))).iloc[0]
    dvp_mult = df_res.get("dvp_mult", df_res.get("DvP Mult", df_res.get("DvP_Mult", pd.Series([None])))).iloc[0]

    # ===============================
    # 📈 MODEL SNAPSHOT — Neon Cards
    # ===============================
    st.markdown("#### 📈 Model Snapshot")

    m1, m2, m3, m4 = st.columns(4)

    # --------------------------------------------------
    # PROJECTION CARD
    # --------------------------------------------------
    with m1:
        try:
            p_val = float(proj) if proj is not None else None
            l_val = float(line) if line is not None else None

            if p_val is not None and l_val is not None:
                if p_val > l_val:
                    arrow = "▲"
                    color = "#22c55e"
                    glow = "rgba(34,197,94,0.45)"
                    bg_grad = "linear-gradient(135deg, rgba(34,197,94,0.20), rgba(34,197,94,0.05))"
                    dir_text = "Higher"
                elif p_val < l_val:
                    arrow = "▼"
                    color = "#ef4444"
                    glow = "rgba(239,68,68,0.45)"
                    bg_grad = "linear-gradient(135deg, rgba(239,68,68,0.20), rgba(239,68,68,0.05))"
                    dir_text = "Lower"
                else:
                    arrow = "▬"
                    color = "#3b82f6"
                    glow = "rgba(59,130,246,0.45)"
                    bg_grad = "linear-gradient(135deg, rgba(59,130,246,0.20), rgba(59,130,246,0.05))"
                    dir_text = "Even"
            else:
                arrow = "▬"
                color = "#64748b"
                glow = "rgba(148,163,184,0.30)"
                bg_grad = "linear-gradient(135deg, rgba(148,163,184,0.18), rgba(148,163,184,0.06))"
                dir_text = "–"
        except Exception:
            arrow = "▬"
            color = "#64748b"
            glow = "rgba(148,163,184,0.30)"
            bg_grad = "linear-gradient(135deg, rgba(148,163,184,0.18), rgba(148,163,184,0.06))"
            dir_text = "–"

        proj_display = "–"
        try:
            if proj is not None:
                proj_display = f"{float(proj):.2f}"
        except Exception:
            pass

        line_display = "-"
        try:
            line_display = f"{float(line):.1f}"
        except Exception:
            pass

        st.markdown(
            f"""
            <div class="metric-card" style="
                padding: 1rem 1.2rem;
                background: {bg_grad};
                border-radius: 15px;
                border: 1px solid rgba(255,255,255,0.12);
                box-shadow: 0 0 18px {glow};
            ">
                <div style="font-size: 0.85rem; opacity: 0.8;">Projection</div>

                <div style="
                    margin-top: 0.25rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 1.45rem;
                    font-weight: 700;
                ">
                    <span>{proj_display}</span>
                    <span style="
                        color:{color};
                        font-size:1.6rem;
                        margin-left:0.5rem;
                        animation: float 1.2s infinite ease-in-out;
                    ">{arrow}</span>
                </div>

                <div style="margin-top: 0.1rem; font-size: 0.9rem; opacity: 0.8;">
                    Line {line_display} · {dir_text}
                </div>
            </div>

            <style>
            @keyframes float {{
                0% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(-3px); }}
                100% {{ transform: translateY(0px); }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------
    # EV CARD
    # --------------------------------------------------
    with m2:
        try:
            ev_val = float(ev_cents) if ev_cents is not None else 0.0
            if ev_val >= 2:
                color = "#22c55e"
                bg_grad = "linear-gradient(135deg, rgba(34,197,94,0.20), rgba(34,197,94,0.05))"
            elif ev_val <= -2:
                color = "#ef4444"
                bg_grad = "linear-gradient(135deg, rgba(239,68,68,0.20), rgba(239,68,68,0.05))"
            else:
                color = "#3b82f6"
                bg_grad = "linear-gradient(135deg, rgba(59,130,246,0.20), rgba(59,130,246,0.05))"
            glow = f"{bg_grad.split('rgba(')[1].split(')')[0]}"
            glow = "rgba(" + glow + ",0.55)" if "rgba" not in glow else "rgba(59,130,246,0.45)"
        except Exception:
            ev_val = 0.0
            color = "#64748b"
            bg_grad = "linear-gradient(135deg, rgba(148,163,184,0.15), rgba(148,163,184,0.05))"
            glow = "rgba(148,163,184,0.30)"

        ev_display = "–"
        try:
            if ev_cents is not None:
                ev_display = f"{float(ev_cents):+,.1f}¢"
        except Exception:
            pass

        st.markdown(
            f"""
            <div class="metric-card" style="
                padding: 1rem 1.2rem;
                background: {bg_grad};
                border-radius: 15px;
                border: 1px solid rgba(255,255,255,0.12);
                box-shadow: 0 0 18px {glow};
            ">
                <div style="font-size: 0.85rem; opacity:0.8;">
                    Expected Value
                </div>

                <div style="font-size:1.45rem;font-weight:700;color:{color};margin-top:0.25rem;">
                    {ev_display}
                </div>

                <div style="font-size:0.9rem; opacity:0.8;">
                    Per $1 exposure
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------
    # MODEL VS BOOK CARD
    # --------------------------------------------------
    with m3:
        try:
            mp = float(model_prob) * 100 if model_prob is not None else None
        except Exception:
            mp = None
        try:
            bp = float(book_prob) * 100 if book_prob is not None else None
        except Exception:
            bp = None

        mp_display = f"{mp:.1f}%" if mp is not None else "–"
        bp_display = f"{bp:.1f}%" if bp is not None else "–"

        st.markdown(
            f"""
            <div class="metric-card" style="
                padding: 1rem 1.2rem;
                background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(15,23,42,0.95));
                border-radius: 15px;
                border: 1px solid rgba(129,140,248,0.35);
                box-shadow: 0 0 18px rgba(129,140,248,0.45);
            ">

                <div style="font-size:0.85rem;opacity:0.8;">
                    Model vs Book
                </div>

                <div style="font-size:1.45rem;font-weight:700;margin-top:0.25rem;">
                    {mp_display}
                </div>

                <div style="font-size:0.9rem;opacity:0.8;">
                    Book implied: {bp_display}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------
    # CONTEXT CARD
    # --------------------------------------------------
    with m4:
        try:
            conf_str = f"{float(confidence)*100:.0f}%"
        except Exception:
            conf_str = "–"

        matchup_bits = []
        if opponent and opponent not in ("–", "", None):
            matchup_bits.append(f"vs {opponent}")
        if position and position not in ("–", "", None):
            matchup_bits.append(str(position))
        try:
            if dvp_mult is not None and dvp_mult != "":
                dvp_val = float(dvp_mult)
                matchup_bits.append(f"DvP {dvp_val:.2f}×")
        except Exception:
            pass
        sub_text = " · ".join(matchup_bits) if matchup_bits else "No matchup data"

        st.markdown(
            f"""
            <div class="metric-card" style="
                padding: 1rem 1.2rem;
                background: linear-gradient(135deg, rgba(250,204,21,0.22), rgba(15,23,42,0.95));
                border-radius: 15px;
                border: 1px solid rgba(250,204,21,0.45);
                box-shadow: 0 0 18px rgba(250,204,21,0.45);
            ">

                <div style="font-size:0.85rem;opacity:0.8;">Context</div>

                <div style="font-size:1.45rem;font-weight:700;margin-top:0.25rem;">
                    {conf_str}
                </div>

                <div style="font-size:0.9rem;opacity:0.8;">
                    {sub_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # ===============================
    # 🧪 RAW RESULT TABLE
    # ===============================
    st.markdown("#### 🔬 Full Result Row")
    st.dataframe(df_res, use_container_width=True)

    # ===============================
    # 📈 Optional distribution chart
    # ===============================
    if "Distribution" in df_res.columns and isinstance(
        df_res["Distribution"].iloc[0], (list, tuple, np.ndarray)
    ):
        try:
            dist_vals = np.array(df_res["Distribution"].iloc[0], dtype=float)
            x = np.arange(len(dist_vals))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=x, y=dist_vals))
            fig.update_layout(
                title="Model Distribution (simulated outcomes)",
                xaxis_title=stat,
                yaxis_title="Probability",
                bargap=0.02,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# ===============================
# 📦 BATCH MODE UI
# ===============================
def init_manual_entries():
    if "manual_entries" not in st.session_state:
        st.session_state.manual_entries = []


def add_manual_entry(player, stat, line, odds):
    st.session_state.manual_entries.append(
        {"Player": player, "Stat": stat, "Line": line, "Odds": odds}
    )


def clear_manual_entries():
    st.session_state.manual_entries = []


def batch_mode_view():
    init_manual_entries()
    render_header()
    st.markdown("### 🧺 Batch Analyzer")

    mode = st.radio(
        "Batch input mode",
        ["Manual entry", "Upload CSV"],
        horizontal=True,
    )

    debug_mode = st.checkbox("Enable debug mode for batch", value=True)

    if mode == "Manual entry":
        st.markdown("#### ✏️ Add props manually")

        with st.form("manual_entry_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns([1.7, 1, 0.8, 0.8])
            with c1:
                m_player = st.text_input("Player", key="batch_player")
            with c2:
                m_stat = st.selectbox(
                    "Stat",
                    ["PTS", "REB", "AST", "PRA", "REB+AST", "PTS+REB", "PTS+AST", "FG3M"],
                    key="batch_stat",
                )
            with c3:
                m_line = st.number_input("Line", key="batch_line", step=0.5, format="%.1f")
            with c4:
                m_odds = st.text_input("Odds", key="batch_odds", value="-110")

            s1, s2 = st.columns([1, 1])
            with s1:
                submitted = st.form_submit_button("➕ Add to slate", use_container_width=True)
            with s2:
                clear_clicked = st.form_submit_button("🧹 Clear all", use_container_width=True)

        if submitted:
            if not m_player.strip():
                st.error("Please enter a valid player name.")
            else:
                add_manual_entry(m_player, m_stat, m_line, m_odds)
                st.success(f"Added {m_player} {m_stat} {m_line} ({m_odds})")

        if clear_clicked:
            clear_manual_entries()
            st.success("Cleared all manual entries.")

        if st.session_state.manual_entries:
            df_preview = pd.DataFrame(st.session_state.manual_entries)
            st.markdown("#### 📋 Current slate")
            st.dataframe(df_preview, use_container_width=True)
        else:
            st.info("No props added yet. Use the form above to build your slate.")

        if st.button("🚀 Analyze batch", type="primary", use_container_width=True):
            if not st.session_state.manual_entries:
                st.error("Add at least one prop before running the batch.")
                return

            df_input = pd.DataFrame(st.session_state.manual_entries)
            with st.spinner("Running batch model…"):
                df_results = run_batch_from_df(df_input, debug_mode=debug_mode)

            if df_results is None:
                return

            if isinstance(df_results, dict):
                df_results = pd.DataFrame([df_results])

            st.markdown("#### 📊 Batch results")
            st.dataframe(df_results, use_container_width=True)

            if "EV" in df_results.columns or "EV¢" in df_results.columns:
                ev_col = "EV" if "EV" in df_results.columns else "EV¢"
                try:
                    df_sorted = df_results.sort_values(by=ev_col, ascending=False)
                    st.markdown("##### 🔝 Highest EV props")
                    st.dataframe(df_sorted.head(25), use_container_width=True)
                except Exception:
                    pass

            to_download = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=to_download,
                file_name=f"proppulse_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.markdown("#### 📂 Upload CSV")

        st.caption(
            "Expected columns (case-insensitive): "
            "`Player`, `Stat`, `Line`, `Odds`."
        )

        file = st.file_uploader("Upload your slate CSV", type=["csv"])

        if file is not None:
            try:
                df_input = pd.read_csv(file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return

            st.markdown("#### 📋 Preview")
            st.dataframe(df_input.head(50), use_container_width=True)

            if st.button("🚀 Analyze uploaded slate", type="primary", use_container_width=True):
                with st.spinner("Running batch model…"):
                    df_results = run_batch_from_df(df_input, debug_mode=debug_mode)

                if df_results is None:
                    return

                if isinstance(df_results, dict):
                    df_results = pd.DataFrame([df_results])

                st.markdown("#### 📊 Batch results")
                st.dataframe(df_results, use_container_width=True)

                if "EV" in df_results.columns or "EV¢" in df_results.columns:
                    ev_col = "EV" if "EV" in df_results.columns else "EV¢"
                    try:
                        df_sorted = df_results.sort_values(by=ev_col, ascending=False)
                        st.markdown("##### 🔝 Highest EV props")
                        st.dataframe(df_sorted.head(25), use_container_width=True)
                    except Exception:
                        pass

                to_download = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results as CSV",
                    data=to_download,
                    file_name=f"proppulse_batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("Upload a CSV file to start batch analysis.")

# ===============================
# 📡 LIVE EV SHEET VIEWER
# ===============================
def live_sheet_view():
    render_header()
    st.markdown("### 📡 Live EV Board (Google Sheets)")

    st.caption(
        "This pulls directly from your live Google Sheet so visitors can see the current EV board "
        "without downloading a file."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🔄 Refresh sheet", use_container_width=True):
            st.session_state["_refresh_sheet"] = datetime.now().timestamp()
    with c2:
        st.write("")

    try:
        df_sheet = pd.read_csv(SHEET_CSV_URL)
    except Exception as e:
        st.error(f"Could not load Google Sheet CSV: {e}")
        return

    if df_sheet.empty:
        st.warning("Sheet loaded, but appears to be empty.")
        return

    important_cols = [
        c for c in df_sheet.columns
        if c.lower() in ["player", "stat", "line", "projection", "ev", "ev¢", "direction", "confidence"]
    ]
    if important_cols:
        st.markdown("#### 🔝 Top EV snapshot")
        preview = df_sheet.copy()
        ev_col = None
        for candidate in ["EV¢", "EV", "ev", "ev¢"]:
            if candidate in preview.columns:
                ev_col = candidate
                break
        if ev_col:
            with pd.option_context("mode.use_inf_as_na", True):
                preview[ev_col] = (
                    preview[ev_col]
                    .astype(str)
                    .str.replace("+", "", regex=False)
                    .str.replace("¢", "", regex=False)
                )
                preview[ev_col] = pd.to_numeric(preview[ev_col], errors="coerce")
            preview = preview.sort_values(by=ev_col, ascending=False)

        st.dataframe(preview[important_cols].head(50), use_container_width=True)

    st.markdown("#### 🧾 Full sheet")
    st.dataframe(df_sheet, use_container_width=True)

# ===============================
# ℹ️ ABOUT VIEW
# ===============================
def about_view():
    render_header()
    st.markdown("### ℹ️ About PropPulse+")

    st.write(
        "PropPulse+ is your calibrated NBA player prop engine. It combines recent form, season-long "
        "context, defense-vs-position multipliers, and matchup-aware logic to surface edges instead "
        "of vibes. The focus is on identifying mispriced lines where role, matchup, and market all align."
    )

    st.write(
        "This app is wired to your underlying Python model in `prop_ev.py`, which handles projections, "
        "distribution fitting, and expected value calculations. The UI is tuned for both mobile and desktop "
        "so you can test single props, run batch slates, and review your live EV sheet from anywhere."
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            Built by <strong>QacePicks</strong> · Powered by <strong>PropPulse+</strong> · v2025.7<br/>
            Data-calibrated · Matchup-aware · EV-first
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===============================
# 🧭 SIDEBAR (Brand + Tips)
# ===============================
with st.sidebar:
    st.markdown("### 🏀 PropPulse+")
    st.caption("QacePicks · PropPulse+ v2025.7")

    st.markdown("---")
    st.caption(
        "Tip: On mobile, use the Streamlit toggle to hide this sidebar and give the tabs more space."
    )

# ===============================
# 🚀 CENTER TABS ROUTER
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Single Prop", "🧺 Batch Mode", "📡 Live EV Sheet", "ℹ️ About"]
)

with tab1:
    single_prop_view()

with tab2:
    batch_mode_view()

with tab3:
    live_sheet_view()

with tab4:
    about_view()
