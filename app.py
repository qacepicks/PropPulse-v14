"""
PropPulse+ v2025.6 — Professional NBA Prop Analyzer
Combined: Analysis Tools + Live Sheet Viewer
Mobile-Optimized UI | Blue–Red Theme | Multi-Tab Interface
"""

import os, io, base64, re
from datetime import datetime, timedelta, timezone
from contextlib import redirect_stdout

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ===============================
# 🔗 Google Sheets config (Live EV Board)
# ===============================
SHEET_ID = "1SHuoEg331k_dcrgBoc7y8gWbgw1QTKHFJRzzNRqiOnE"
SHEET_GID = "1954146299"  # the worksheet/tab you're using
SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
    f"?format=csv&gid={SHEET_GID}"
)

# =============================
# 🔒 ADMIN CONFIG (Private Use)
# =============================
ADMIN_CODE = "qace"

# ============
# Model import
# ============
try:
    import prop_ev as pe
except ImportError as e:
    st.error(f"❌ Failed to import prop_ev.py: {e}")
    st.stop()

# =============
# Page settings
# =============
st.set_page_config(
    page_title="PropPulse+ | NBA Props",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🔗 LOCAL DATA CONFIG (for Live Slate manual upload)
DATA_DIR = "data"
LATEST_FILE = os.path.join(DATA_DIR, "latest_props.xlsx")


def cleanup_old_results():
    """Keep only latest_props.xlsx and all player logs (CSV); delete older .xlsx results."""
    if not os.path.exists(DATA_DIR):
        return
    for f in os.listdir(DATA_DIR):
        if f.endswith(".xlsx") and f != "latest_props.xlsx":
            try:
                os.remove(os.path.join(DATA_DIR, f))
            except Exception:
                pass


# =========
# Logo util
# =========
def _logo_b64():
    path = "proppulse_logo.png"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


logo_b64 = _logo_b64()
logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="width:56px;height:56px;border-radius:12px;">'
    if logo_b64
    else '<div class="brand-logo-fallback">PP</div>'
)

# ======================
# Global color variables
# ======================
PRIMARY = "#2563EB"
PRIMARY_DARK = "#1E40AF"
PRIMARY_LIGHT = "#60A5FA"
ACCENT = "#EF4444"
ACCENT_DARK = "#B91C1C"
TEXT_PRIMARY = "#F9FAFB"
TEXT_SECONDARY = "#E5E7EB"
TEXT_MUTED = "#9CA3AF"
SURFACE = "#111827"
SURFACE_2 = "#1F2937"
BORDER = "#374151"

# =========
# Global CSS
# =========
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {{
  --primary:{PRIMARY};
  --primary-dark:{PRIMARY_DARK};
  --primary-light:{PRIMARY_LIGHT};
  --accent:{ACCENT};
  --accent-dark:{ACCENT_DARK};
  --bg:{SURFACE};
  --surface:{SURFACE_2};
  --text:{TEXT_PRIMARY};
  --text-2:{TEXT_SECONDARY};
  --muted:{TEXT_MUTED};
  --border:{BORDER};
}}

.stApp {{
  background: var(--bg);
  font-family: 'Inter', sans-serif;
  color: var(--text-2);
  overflow-x: hidden !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}

[data-testid="stSidebar"] {{
  background: var(--surface);
  border-right: 2px solid var(--primary);
  overflow-y: auto;
}}
[data-testid="stSidebar"] * {{ color: var(--text-2) !important; }}
[data-testid="stSidebar"] label {{
  font-weight: 700; font-size: 12px; text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 6px;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 8px;
  background: var(--surface);
  border-radius: 12px;
  padding: 8px;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  border-radius: 8px;
  color: var(--text-2);
  font-weight: 600;
  padding: 12px 24px;
}}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: white !important;
}}

.main-header {{
  background: linear-gradient(135deg, var(--surface), var(--bg));
  border-bottom: 2px solid var(--primary);
  padding: 1.2rem 1rem; box-shadow: 0 8px 24px rgba(0,0,0,.4);
}}
.brand-container {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }}
.brand-logo-fallback {{
  width: 56px; height: 56px; background: linear-gradient(135deg, var(--primary), var(--accent));
  border-radius: 12px; display: flex; align-items: center; justify-content: center;
  font-weight: 900; font-size: 26px; color: white;
}}
.brand-title {{
  font-size: 28px; font-weight: 900;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.brand-subtitle {{ font-size: 13px; color: var(--muted); }}
.status-badge {{
  background: rgba(16, 185, 129, .12); border: 2px solid #10B981; color: #10B981;
  padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 700;
}}

.stTextInput input, .stNumberInput input, .stSelectbox select {{
  background: var(--surface); border: 2px solid var(--border);
  border-radius: 10px; color: var(--text); padding: 12px 14px; font-size: 15px;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {{
  border-color: var(--primary);
  box-shadow: 0 0 8px rgba(37, 99, 235, 0.35);
  outline: none;
}}

[data-testid="stDateInput"] input {{
  background-color: var(--surface) !important;
  border: 2px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
  padding: 10px 12px !important;
  font-size: 15px !important;
}}
[data-testid="stDateInput"] label {{
  font-weight: 700 !important; font-size: 12px !important; text-transform: uppercase !important;
  letter-spacing: 1px !important; margin-bottom: 6px !important; color: var(--text-2) !important;
}}
[data-testid="stDateInput"] input:hover, [data-testid="stDateInput"] input:focus {{
  border-color: var(--primary) !important;
  box-shadow: 0 0 8px rgba(37, 99, 235, 0.35) !important;
  outline: none !important; transition: 0.25s ease-in-out !important;
}}

.stButton>button {{
  width: 100%;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  border-radius: 10px; padding: 14px 24px; font-weight: 800; text-transform: uppercase;
  box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
  color: white;
}}
.stButton>button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(239, 68, 68, 0.45);
}}

.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 24px 0; }}
.metric-card {{
  background: var(--surface); border-radius: 12px; border: 1.5px solid var(--border);
  padding: 18px; transition: .3s;
}}
.metric-card:hover {{ border-color: var(--primary); transform: translateY(-3px); }}
.metric-value {{ font-size: 28px; font-weight: 900; color: var(--text); }}

.ev-container {{
  margin-top: 12px; background: linear-gradient(135deg, rgba(37,99,235,.08), rgba(239,68,68,.08));
  border: 1px solid var(--border); border-radius: 12px; padding: 16px;
}}
.ev-badge {{ display: inline-block; font-weight: 900; padding: 6px 10px; border-radius: 8px; margin-right: 10px; }}
.ev-positive {{ background: rgba(16, 185, 129, .15); color: #10B981; border: 1px solid #10B981; }}
.ev-negative {{ background: rgba(239, 68, 68, .15); color: var(--accent); border: 1px solid var(--accent); }}
.recommendation {{ display: inline-block; font-weight: 800; color: var(--text-2); }}

.footer {{
  text-align:center; padding:30px 0; font-size:13px; color: var(--muted);
  border-top:1px solid var(--border); margin-top:40px;
}}
.footer strong {{ color: var(--primary) !important; }}

@media (max-width: 768px) {{
  .brand-title {{ font-size: 22px; }}
  .metric-grid {{ grid-template-columns: 1fr; }}
  .stAppViewContainer {{ padding: 0 .5rem !important; }}
  .stButton>button {{ font-size: 14px; padding: 12px 20px; }}
  .ev-container {{ padding: 16px; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ======
# Header
# ======
def render_header(is_admin: bool = False):
    """Render header with optional Admin badge."""
    admin_badge_html = ""
    if is_admin:
        admin_badge_html = """
        <div class="status-badge" style="background: rgba(16,185,129,.15);
             border: 2px solid #10B981; color:#10B981; margin-left:10px;">
             🧠 ADMIN MODE
        </div>
        """

    st.markdown(
        f"""
        <div class="main-header">
          <div class="brand-container">
            {logo_html}
            <div class="brand-text">
              <div class="brand-title">PropPulse+</div>
              <div class="brand-subtitle">Advanced NBA Player Prop Analytics Platform</div>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
              <div class="status-badge">LIVE</div>
              {admin_badge_html}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# HELPER FUNCTIONS FOR LIVE SHEET
# ==========================================
def to_csv_url(url: str, gid: int | None = None) -> str:
    """Convert Google Sheets publish or edit URL to CSV endpoint."""
    if "output=csv" in url:
        return url
    if "pubhtml" in url:
        base = url.split("/pubhtml")[0]
        if gid is not None:
            return f"{base}/pub?gid={gid}&single=true&output=csv"
        return f"{base}/pub?output=csv"
    if "docs.google.com/spreadsheets" in url and "output=" not in url:
        sep = "&" if "?" in url else "?"
        if gid is not None and "gid=" not in url:
            url = f"{url}{sep}gid={gid}"
            sep = "&"
        return f"{url}{sep}output=csv"
    return url


@st.cache_data(ttl=120, show_spinner=False)
def load_sheet(sheet_url: str, gid: int | None = None) -> pd.DataFrame:
    """Load Google Sheet, try CSV first then HTML fallback."""
    csv_url = to_csv_url(sheet_url, gid)
    try:
        df = pd.read_csv(csv_url)
    except Exception:
        try:
            tables = pd.read_html(sheet_url)
            if not tables:
                raise ValueError("No tables found in published sheet HTML")
            df = tables[0]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Google Sheet. Error: {e}")

    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # Column renames to normalize your sheet format
    ren = {
        "Ev": "EV",
        "Edge": "EV",
        "EV¢": "EV",  # your sheet uses EV¢
        "Confidence %": "Confidence",
        "Conf": "Confidence",
        "Opponent Team": "Opponent",
        "Games": "Games Analyzed",
        "GamesAnalyzed": "Games Analyzed",
    }
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert specified columns to numeric."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize_results(df: pd.DataFrame, result_col: str = "Result") -> dict:
    """Count results and compute win rate."""
    if result_col not in df.columns:
        return {"wins": 0, "losses": 0, "close": 0, "total": 0, "win_rate": 0.0}
    series = df[result_col].astype(str).str.strip()
    wins = (series == "✓").sum()
    losses = (series == "✗").sum()
    close = (series == "⚠️").sum() + (series == "⚠").sum()
    total = wins + losses + close
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    return {"wins": wins, "losses": losses, "close": close, "total": total, "win_rate": win_rate}


def drop_alt_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out alternate lines (0.5, 1.5, 2.5, etc.)."""
    if "Line" not in df.columns:
        return df

    def is_half(x):
        try:
            val = float(x)
            frac = abs(val - round(val))
            return abs(frac - 0.5) < 1e-9
        except Exception:
            return False

    mask = ~df["Line"].apply(is_half)
    return df[mask]


def color_ev(val):
    """Style function for EV column."""
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 0:
        return "background-color: rgba(52,199,89,0.12); color: #caf7d2;"
    if v < 0:
        return "background-color: rgba(229,83,83,0.10); color: #ffd6d6;"
    return ""


# =================
# MAIN TAB SELECTOR
# =================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 Analysis Tools",
        "💎 Live Slate",
        "📊 Batch Analysis",
        "📁 CSV Import",
        "⚖️ Prop Comparison",
    ]
)

# ==========================================
# TAB 1: ANALYSIS TOOLS
# ==========================================
with tab1:
    render_header()

    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        st.markdown("---")

        with st.expander("🔧 Advanced Settings"):
            debug_mode = st.checkbox("Enable Debug Mode", value=True)
            show_charts = st.checkbox("Show Visualizations", value=True)

    with st.form("prop_analyzer"):
        st.markdown("### 📋 Enter Prop Details")

        c1, c2 = st.columns([2, 1])
        player = c1.text_input("Player Name", placeholder="LeBron James")
        stat = c2.selectbox(
            "Stat Category",
            ["PTS", "REB", "AST", "REB+AST", "PRA", "P+R", "P+A", "FG3M"],
        )

        cdate = st.columns([1])[0]
        analysis_date = cdate.date_input(
            "📅 Analysis Date",
            value=datetime.now(),
            min_value=datetime.now() - timedelta(days=7),
            max_value=datetime.now() + timedelta(days=7),
            help="Select the date for opponent/schedule lookup.",
        )

        c3, c4 = st.columns(2)
        line = c3.number_input("Line", 0.0, 100.0, 25.5, 0.5)
        odds = c4.number_input("Odds (American)", value=-110, step=5)

        st.markdown("---")
        submitted = st.form_submit_button("🔍 ANALYZE PROP", use_container_width=True)

    if submitted:
        if not player.strip():
            st.error("⚠️ Please enter a player name")
            st.stop()

        try:
            settings = pe.load_settings()
        except Exception as e:
            st.error(f"❌ Failed to load settings: {e}")
            st.stop()

        analysis_date_str = analysis_date.strftime("%Y-%m-%d")
        st.info(f"📅 Analyzing for date: **{analysis_date_str}**")

        with st.spinner(f"🏀 Analyzing {player}'s {stat} projection..."):
            try:
                settings["analysis_date"] = analysis_date_str
                buf = io.StringIO()
                with redirect_stdout(buf):
                    result = pe.analyze_single_prop(
                        player,
                        stat,
                        float(line),
                        int(odds),
                        settings=settings,
                        debug_mode=debug_mode,
                    )
                debug_text = buf.getvalue()
                if not result:
                    st.error("❌ Unable to analyze this prop.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Analysis Error: {e}")
                st.stop()

        p_model = result["p_model"]
        p_book = result["p_book"]
        ev = result["ev"]
        projection = result["projection"]
        n_games = result["n_games"]
        opponent = result.get("opponent", "N/A")
        position = result.get("position", "N/A")
        dvp_mult = result.get("dvp_mult", 1.0)
        confidence = result.get("confidence", 0.0)
        grade = result.get("grade", "N/A")

        edge = (p_model - p_book) * 100
        ev_cents = ev * 100
        recommendation = "OVER" if projection > line else "UNDER"

        st.success("✅ Analysis Complete!")

        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(
            f"<div class='metric-card'><div class='metric-value'>{projection:.1f}</div><div>Model Projection</div></div>",
            unsafe_allow_html=True,
        )
        mc2.markdown(
            f"<div class='metric-card'><div class='metric-value'>{ev_cents:+.1f}¢</div><div>Expected Value</div></div>",
            unsafe_allow_html=True,
        )
        mc3.markdown(
            f"<div class='metric-card'><div class='metric-value'>{edge:+.1f}%</div><div>Model Edge</div></div>",
            unsafe_allow_html=True,
        )
        mc4.markdown(
            f"<div class='metric-card'><div class='metric-value'>{confidence:.2f}</div><div>Confidence</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        ev_class = "ev-positive" if ev > 0 else "ev-negative"
        rec_text = f"BET {recommendation}" if ev > 0 else "FADE THIS PROP"
        st.markdown(
            f"<div class='ev-container'><span class='ev-badge {ev_class}'>{ev_cents:+.1f}¢ EV</span>"
            f"<span class='recommendation'>{rec_text}</span></div>",
            unsafe_allow_html=True,
        )

        cL, cR = st.columns(2)
        cL.markdown(
            f"**Model Prob:** {p_model*100:.1f}%  \n"
            f"**Book Prob:** {p_book*100:.1f}%  \n"
            f"**Line:** {line}  \n"
            f"**Odds:** {odds}"
        )
        cR.markdown(
            f"**Opponent:** {opponent}  \n"
            f"**Position:** {position}  \n"
            f"**DvP Multiplier:** {dvp_mult:.3f}  \n"
            f"**Sample Size:** {n_games}"
        )

        if show_charts:
            st.markdown("---")
            st.markdown("### 📈 Visual Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🎯 Win Probability Comparison")
                fig1 = go.Figure()

                fig1.add_trace(
                    go.Bar(
                        y=["Our Model", "Sportsbook"],
                        x=[p_model * 100, p_book * 100],
                        orientation="h",
                        marker=dict(
                            color=[
                                "#10B981" if p_model > p_book else "#EF4444",
                                "#6B7280",
                            ],
                            line=dict(width=0),
                        ),
                        text=[f"{p_model*100:.1f}%", f"{p_book*100:.1f}%"],
                        textposition="inside",
                        textfont=dict(size=16, color="white", family="Inter"),
                        hovertemplate="%{x:.1f}%<extra></extra>",
                    )
                )

                fig1.update_layout(
                    template="plotly_dark",
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(
                        range=[0, 100],
                        title="Win Probability (%)",
                        showgrid=False,
                    ),
                    yaxis=dict(showgrid=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )

                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown("#### 💰 Expected Value")
                ev_color = "#10B981" if ev_cents > 0 else "#EF4444"
                ev_text = "POSITIVE" if ev_cents > 0 else "NEGATIVE"

                fig2 = go.Figure()

                fig2.add_trace(
                    go.Indicator(
                        mode="number+delta",
                        value=ev_cents,
                        title={
                            "text": f"<b>{ev_text} EV</b>",
                            "font": {"size": 18, "color": "#E5E7EB"},
                        },
                        number={
                            "suffix": "¢",
                            "font": {
                                "size": 48,
                                "color": ev_color,
                                "family": "Inter",
                            },
                        },
                        delta={
                            "reference": 0,
                            "relative": False,
                            "position": "bottom",
                        },
                        domain={"x": [0, 1], "y": [0, 1]},
                    )
                )

                fig2.update_layout(
                    template="plotly_dark",
                    height=200,
                    margin=dict(l=10, r=10, t=40, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### 📊 Your Edge vs The Market")
            fig3 = go.Figure()

            edge_color = "#10B981" if edge > 0 else "#EF4444"

            fig3.add_trace(
                go.Bar(
                    x=["Market Edge"],
                    y=[edge],
                    marker=dict(color=edge_color, line=dict(width=0)),
                    text=[f"{edge:+.1f}%"],
                    textposition="outside",
                    textfont=dict(size=20, color=edge_color, family="Inter"),
                    hovertemplate="Edge: %{y:+.1f}%<extra></extra>",
                    width=0.4,
                )
            )

            fig3.add_hline(
                y=0, line_dash="dash", line_color="#6B7280", line_width=2
            )

            fig3.update_layout(
                template="plotly_dark",
                height=250,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(
                    title="Edge (%)",
                    showgrid=True,
                    gridcolor="rgba(107, 114, 128, 0.2)",
                ),
                xaxis=dict(showticklabels=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )

            st.plotly_chart(fig3, use_container_width=True)

        if debug_mode and debug_text:
            with st.expander("🔧 Debug Log"):
                st.code(debug_text)

# ==========================================
# TAB 2: LIVE SLATE (GOOGLE SHEETS)
# ==========================================
with tab2:
    # Admin zone
    with st.expander("📤 Admin Upload (Private)", expanded=False):
        admin_code = st.text_input(
            "Enter admin code to enable upload:", type="password", key="admin_code"
        )
        is_admin = admin_code == ADMIN_CODE

        if is_admin:
            st.success("✅ Admin access granted — you can now upload a new sheet.")
            uploaded_file = st.file_uploader(
                "Upload new latest_props.xlsx",
                type=["xlsx"],
                key="admin_upload",
            )
            if uploaded_file:
                try:
                    cleanup_old_results()
                    os.makedirs(DATA_DIR, exist_ok=True)
                    file_path = os.path.join(DATA_DIR, "latest_props.xlsx")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("✅ Sheet updated successfully. Refreshing dashboard...")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save uploaded file: {e}")

            if st.button("🚪 Log Out of Admin Mode", use_container_width=True):
                st.session_state.pop("admin_code", None)
                st.success("🔒 Logged out of admin mode.")
                st.rerun()
        elif admin_code:
            is_admin = False
            st.error("❌ Invalid admin code.")
        else:
            is_admin = False

    # Live Slate header
    render_header(is_admin=is_admin)
    st.markdown("### 💎 Today's Live Slate")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(
            f"<p style='color:{TEXT_MUTED};font-size:14px;'>📡 Auto-synced from Google Sheets</p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<p style='color:{TEXT_MUTED};font-size:14px;'>🕐 Last updated: "
            f"{datetime.now(tz=timezone.utc).strftime('%I:%M %p UTC')}</p>",
            unsafe_allow_html=True,
        )
    with col3:
        if st.button("⚡ Refresh", use_container_width=True, key="refresh_slate"):
            load_sheet.clear()
            st.rerun()

    # You chose option C: edit link; we pass it through load_sheet → to_csv_url
    SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSUhlkHIhBOhxyDZafRM0-7S933fm1-8bC6Gj6219MH4EW0thdz1YLSpBIB9OzCiKQ-Rm76TKUZ9CSp/pub?output=csv"

    with st.spinner("📊 Loading live sheet data..."):
        try:
            df_raw = load_sheet(SHEET_URL)
        except Exception as e:
            st.error(f"❌ Failed to load Google Sheet: {e}")
            st.stop()

    # ===== DATA CLEANUP & STATS =====
    num_candidates = [
        "Line",
        "Projection",
        "EV",
        "Confidence",
        "Odds",
        "Games Analyzed",
        "DvP Mult",
    ]
    df_raw = coerce_numeric(df_raw, num_candidates)

    sum_stats_all = summarize_results(df_raw, result_col="Result")
    total_props = len(df_raw)
    if "EV" in df_raw.columns:
        positive_ev_count = len(df_raw[df_raw["EV"].fillna(-9999) > 0])
    else:
        positive_ev_count = 0

    st.markdown("---")

    st1, st2, st3, st4, st5 = st.columns(5)
    st1.markdown(
        f"""
<div class='metric-card' style='text-align: center;'>
    <div style='font-size: 24px; font-weight: 900; color: {PRIMARY};'>{total_props}</div>
    <div style='font-size: 12px; color: {TEXT_MUTED};'>TOTAL PROPS</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st2.markdown(
        f"""
<div class='metric-card' style='text-align: center;'>
    <div style='font-size: 24px; font-weight: 900; color: #10B981;'>{positive_ev_count}</div>
    <div style='font-size: 12px; color: {TEXT_MUTED};'>+EV PLAYS</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st3.markdown(
        f"""
<div class='metric-card' style='text-align: center;'>
    <div style='font-size: 24px; font-weight: 900; color: #10B981;'>{sum_stats_all['wins']}</div>
    <div style='font-size: 12px; color: {TEXT_MUTED};'>WINS ✓</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st4.markdown(
        f"""
<div class='metric-card' style='text-align: center;'>
    <div style='font-size: 24px; font-weight: 900; color: {ACCENT};'>{sum_stats_all['losses']}</div>
    <div style='font-size: 12px; color: {TEXT_MUTED};'>LOSSES ✗</div>
</div>
""",
        unsafe_allow_html=True,
    )

    win_rate_color = "#10B981" if sum_stats_all["win_rate"] >= 50 else ACCENT
    st5.markdown(
        f"""
<div class='metric-card' style='text-align: center;'>
    <div style='font-size: 24px; font-weight: 900; color: {win_rate_color};'>{sum_stats_all['win_rate']:.1f}%</div>
    <div style='font-size: 12px; color: {TEXT_MUTED};'>WIN RATE</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ===== FILTERS, TABLE, SUMMARY =====
    with st.expander("🎛️ Filter Props", expanded=False):
        st.markdown("#### Quick Filters")
        qf1, qf2, qf3 = st.columns(3)
        with qf1:
            only_top = st.checkbox("✨ Only Positive EV", value=False)
        with qf2:
            hide_alts = st.checkbox("🎯 Hide Alt Lines (.5)", value=False)
        with qf3:
            min_edge = st.number_input(
                "💰 Min EV", value=0.0, step=0.5, help="Minimum expected value in cents"
            )

        st.markdown("#### Search & Filter")
        f1, f2 = st.columns(2)
        with f1:
            search = st.text_input(
                "🔍 Search Player Name", placeholder="Type player name..."
            )
        with f2:
            min_conf = st.number_input(
                "📊 Min Confidence",
                value=0.0,
                step=5.0,
                help="Minimum confidence percentage",
            )

        f3, f4 = st.columns(2)
        with f3:
            stat_opt = sorted(
                [
                    s
                    for s in df_raw.get("Stat", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .unique()
                ]
            )
            stat_sel = st.multiselect(
                "📈 Stat Type",
                stat_opt,
                default=[],
                placeholder="Filter by stat...",
            )
        with f4:
            opp_opt = sorted(
                [
                    s
                    for s in df_raw.get("Opponent", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .unique()
                ]
            )
            opp_sel = st.multiselect(
                "🏀 Opponent Team",
                opp_opt,
                default=[],
                placeholder="Filter by opponent...",
            )

        st.markdown("---")
        st.caption(
            "💡 Confidence measures how consistent the model’s data is (higher = more reliable)."
        )

    filtered = df_raw.copy()

    if hide_alts:
        filtered = drop_alt_lines(filtered)
    if search:
        filtered = filtered[
            filtered["Player"].astype(str).str.contains(
                search, case=False, na=False
            )
        ]
    if stat_sel:
        filtered = filtered[filtered["Stat"].astype(str).isin(stat_sel)]
    if opp_sel and "Opponent" in filtered.columns:
        filtered = filtered[filtered["Opponent"].astype(str).isin(opp_sel)]
    if "EV" in filtered.columns:
        filtered = filtered[filtered["EV"].fillna(-9999) >= float(min_edge)]
    if "Confidence" in filtered.columns:
        filtered = filtered[
            filtered["Confidence"].fillna(-9999) >= float(min_conf)
        ]
    if only_top and "EV" in filtered.columns:
        filtered = filtered[filtered["EV"].fillna(-9999) > 0]

    if len(filtered) != len(df_raw):
        st.info(
            f"📌 Showing **{len(filtered)}** of **{len(df_raw)}** props after filters"
        )

    def style_row(row):
        styles = [""] * len(row)
        if "EV" in filtered.columns:
            ev_idx = list(filtered.columns).index("EV")
            try:
                ev_val = float(row.iloc[ev_idx])
                if ev_val > 0:
                    styles = [
                        "background-color: rgba(16,185,129,0.08); "
                        "border-left: 3px solid #10B981;"
                    ] * len(row)
                elif ev_val < -5:
                    styles = [
                        f"background-color: rgba(239,68,68,0.08); "
                        f"border-left: 3px solid {ACCENT};"
                    ] * len(row)
            except Exception:
                pass
        return styles

    styled = filtered.copy()
    if len(styled) > 0:
        styled_show = styled.style.apply(style_row, axis=1)
        if "EV" in styled.columns:
            styled_show = styled_show.apply(
                lambda col: [color_ev(v) for v in col], subset=["EV"]
            )
        st.markdown("### 📋 Props Table")
        st.dataframe(styled_show, use_container_width=True, height=1200)
    else:
        st.warning(
            "⚠️ No props match your current filters. Try adjusting the criteria above."
        )

    sum_stats = summarize_results(filtered, result_col="Result")

    if len(filtered) > 0:
        st.markdown("---")
        st.markdown("### 📊 Filtered Results Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Wins", f"✓ {sum_stats['wins']}")
        c2.metric("Losses", f"✗ {sum_stats['losses']}")
        c3.metric("Close Calls", f"⚠️ {sum_stats['close']}")
        c4.metric("Total Tracked", f"{sum_stats['total']}")
        win_rate_delta = (
            sum_stats["win_rate"] - 50.0 if sum_stats["total"] > 0 else 0
        )
        c5.metric(
            "Win Rate",
            f"{sum_stats['win_rate']:.1f}%",
            delta=f"{win_rate_delta:+.1f}%",
        )
        st.markdown("---")

# ==========================================
# TAB 3: BATCH ANALYSIS (Multi-Prop)
# ==========================================
with tab3:
    st.markdown("### 📊 Batch Prop Analyzer")
    st.caption(
        "Upload a CSV or Excel file containing player props to analyze multiple lines automatically."
    )

    uploaded_batch = st.file_uploader(
        "📤 Upload File (.csv or .xlsx)",
        type=["csv", "xlsx"],
        key="batch_upload",
    )

    if uploaded_batch is not None:
        try:
            if uploaded_batch.name.endswith(".csv"):
                df_batch = pd.read_csv(uploaded_batch)
            else:
                df_batch = pd.read_excel(uploaded_batch, engine="openpyxl")

            st.success(f"✅ Loaded {len(df_batch)} rows successfully.")
            st.dataframe(df_batch.head(10), use_container_width=True)

            if st.button("🚀 Analyze Batch", use_container_width=True):
                with st.spinner("Analyzing all props... please wait ⏳"):
                    results = []
                    settings = pe.load_settings()
                    for _, row in df_batch.iterrows():
                        try:
                            player = str(row.get("Player", "")).strip()
                            stat = str(row.get("Stat", "")).strip()
                            line = float(row.get("Line", 0))
                            odds = int(row.get("Odds", -110))
                            res = pe.analyze_single_prop(
                                player, stat, line, odds, settings=settings
                            )
                            if res:
                                results.append(res)
                        except Exception as e:
                            st.warning(f"⚠️ Skipped one row ({e})")
                            continue

                    if len(results) == 0:
                        st.error("❌ No valid results.")
                        st.stop()

                    df_results = pd.DataFrame(results)
                    st.success(
                        f"✅ Completed batch analysis of {len(df_results)} props!"
                    )
                    st.dataframe(
                        df_results, use_container_width=True, height=500
                    )

                    csv_buf = io.BytesIO()
                    df_results.to_csv(csv_buf, index=False)
                    st.download_button(
                        "💾 Download Results as CSV",
                        data=csv_buf.getvalue(),
                        file_name="propulse_batch_results.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"❌ Failed to read uploaded file: {e}")
    else:
        st.info("📎 Upload a batch file above to start.")

# ==========================================
# TAB 4: CSV IMPORT / VIEWER
# ==========================================
with tab4:
    st.markdown("### 📁 Import or Preview Any Local CSV File")

    file = st.file_uploader(
        "Select a CSV or Excel file", type=["csv", "xlsx"]
    )
    if file:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, engine="openpyxl")
            st.success(f"✅ Loaded {len(df)} rows from {file.name}")
            st.dataframe(
                df.head(50), use_container_width=True, height=400
            )
        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
    else:
        st.info("Upload a file to preview its contents.")

# ==========================================
# TAB 5: PROP COMPARISON TOOL
# ==========================================
with tab5:
    st.markdown("### ⚖️ Prop Comparison — Compare Two Player Lines")

    c1, c2 = st.columns(2)
    with c1:
        p1 = st.text_input("Player 1", "LeBron James")
        s1 = st.selectbox(
            "Stat 1", ["PTS", "REB", "AST", "PRA", "FG3M"], key="s1"
        )
        l1 = st.number_input(
            "Line 1", 0.0, 100.0, 25.5, 0.5, key="l1"
        )
    with c2:
        p2 = st.text_input("Player 2", "Jayson Tatum")
        s2 = st.selectbox(
            "Stat 2", ["PTS", "REB", "AST", "PRA", "FG3M"], key="s2"
        )
        l2 = st.number_input(
            "Line 2", 0.0, 100.0, 25.5, 0.5, key="l2"
        )

    if st.button("🔍 Compare", use_container_width=True):
        try:
            settings = pe.load_settings()
            r1 = pe.analyze_single_prop(p1, s1, l1, -110, settings=settings)
            r2 = pe.analyze_single_prop(p2, s2, l2, -110, settings=settings)

            comp_df = pd.DataFrame(
                [
                    {
                        "Player": p1,
                        "Stat": s1,
                        "Projection": r1["projection"],
                        "EV¢": round(r1["ev"] * 100, 1),
                    },
                    {
                        "Player": p2,
                        "Stat": s2,
                        "Projection": r2["projection"],
                        "EV¢": round(r2["ev"] * 100, 1),
                    },
                ]
            )
            st.dataframe(comp_df, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=comp_df["Player"],
                    y=comp_df["EV¢"],
                    marker=dict(color=["#10B981", "#EF4444"]),
                    text=comp_df["EV¢"],
                    textposition="outside",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                yaxis_title="Expected Value (¢)",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Comparison failed: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    """
    <div class="footer">
      © 2025 <strong>PropPulse+</strong> — Developed by <strong>QacePicks</strong>.<br>
      Data-calibrated | Matchup-weighted | Automated NBA Prop Analytics
    </div>
    """,
    unsafe_allow_html=True,
)
