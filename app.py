"""
PropPulse+ v2025.5 — Professional NBA Prop Analyzer
Modern UI with custom branding and enhanced visualizations
"""

import streamlit as st
import pandas as pd
import os
import base64
from datetime import datetime
import io
from contextlib import redirect_stdout
import plotly.graph_objects as go
import plotly.express as px

# Import your model
try:
    import prop_ev as pe
except ImportError as e:
    st.error(f"❌ Failed to import prop_ev.py: {e}")
    st.stop()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="PropPulse+ | NBA Props",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOGO HANDLING
# ==========================================
def get_logo_base64():
    """Convert logo to base64 for embedding"""
    logo_path = "proppulse_logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# ==========================================
# CUSTOM CSS - Professional Modern UI
# ==========================================
logo_b64 = get_logo_base64()
logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width: 56px; height: 56px; border-radius: 12px;">' if logo_b64 else '<div class="brand-logo-fallback">PP</div>'

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Root Variables for Easy Customization */
    :root {{
        --primary-color: #FF8C42;
        --primary-dark: #E67A33;
        --primary-light: #FFB380;
        --secondary-color: #3B82F6;
        --background-dark: #0B0F19;
        --surface-dark: #1F2937;
        --surface-light: #374151;
        --text-primary: #F9FAFB;
        --text-secondary: #E5E7EB;
        --text-muted: #9CA3AF;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --border-color: #374151;
    }}
    
    /* Global Styles */
    .stApp {{
        background: var(--background-dark);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Force text contrast */
    .stApp, .stMarkdown, p, span, label, div {{
        color: var(--text-secondary) !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Professional Header */
    .main-header {{
        background: linear-gradient(135deg, var(--surface-dark) 0%, var(--background-dark) 100%);
        border-bottom: 2px solid var(--primary-color);
        padding: 2rem 2rem;
        margin: -6rem -2rem 2rem -2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }}
    
    .brand-container {{
        display: flex;
        align-items: center;
        gap: 1.25rem;
    }}
    
    .brand-logo-fallback {{
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 900;
        color: white;
        box-shadow: 0 4px 16px rgba(255, 140, 66, 0.5);
    }}
    
    .brand-text {{
        flex: 1;
    }}
    
    .brand-title {{
        font-size: 32px;
        font-weight: 900;
        color: var(--text-primary);
        letter-spacing: -1px;
        margin: 0;
        line-height: 1;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .brand-subtitle {{
        font-size: 14px;
        color: var(--text-muted);
        margin-top: 6px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    
    .status-badge {{
        background: linear-gradient(135deg, #065F46 0%, #047857 100%);
        color: var(--success);
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 700;
        border: 2px solid var(--success);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .status-badge::before {{
        content: '';
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: var(--surface-dark);
        border-right: 2px solid var(--primary-color);
    }}
    
    [data-testid="stSidebar"] * {{
        color: var(--text-secondary) !important;
    }}
    
    /* Form Controls */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {{
        background: var(--surface-dark) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {{
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 4px rgba(255, 140, 66, 0.15) !important;
    }}
    
    /* Labels */
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label {{
        color: var(--text-muted) !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 8px !important;
    }}
    
    /* Primary Button */
    .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 18px 28px;
        font-weight: 800;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 6px 20px rgba(255, 140, 66, 0.4);
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
        box-shadow: 0 8px 28px rgba(255, 140, 66, 0.6);
        transform: translateY(-2px);
    }}
    
    /* Metric Cards */
    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 20px;
        margin: 28px 0;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, var(--surface-dark) 0%, #151b28 100%);
        border: 2px solid var(--border-color);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
    }}
    
    .metric-card:hover {{
        border-color: var(--primary-color);
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.3);
    }}
    
    .metric-label {{
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }}
    
    .metric-value {{
        color: var(--text-primary);
        font-size: 36px;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 10px;
    }}
    
    .metric-sublabel {{
        font-size: 13px;
        font-weight: 500;
        margin-top: 10px;
    }}
    
    .metric-positive {{
        color: var(--success);
    }}
    
    .metric-negative {{
        color: var(--danger);
    }}
    
    .metric-neutral {{
        color: var(--text-muted);
    }}
    
    /* EV Display */
    .ev-container {{
        text-align: center;
        padding: 40px;
        background: linear-gradient(135deg, var(--surface-dark) 0%, #151b28 100%);
        border: 3px solid var(--border-color);
        border-radius: 20px;
        margin: 28px 0;
        position: relative;
        overflow: hidden;
    }}
    
    .ev-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at top right, rgba(255, 140, 66, 0.1), transparent);
        pointer-events: none;
    }}
    
    .ev-badge {{
        display: inline-block;
        padding: 20px 40px;
        border-radius: 16px;
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 20px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
        position: relative;
        z-index: 1;
    }}
    
    .ev-positive {{
        background: linear-gradient(135deg, #059669 0%, var(--success) 100%);
        color: white;
        border: 3px solid var(--success);
    }}
    
    .ev-negative {{
        background: linear-gradient(135deg, #DC2626 0%, var(--danger) 100%);
        color: white;
        border: 3px solid var(--danger);
    }}
    
    .recommendation {{
        color: var(--text-primary);
        font-size: 22px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
    }}
    
    /* Info Cards */
    .info-card {{
        background: linear-gradient(135deg, var(--surface-dark) 0%, #151b28 100%);
        border: 2px solid var(--border-color);
        border-left: 4px solid var(--primary-color);
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        transition: all 0.2s ease;
    }}
    
    .info-card:hover {{
        border-left-color: var(--primary-light);
        box-shadow: 0 4px 16px rgba(255, 140, 66, 0.2);
    }}
    
    .info-card strong {{
        color: var(--primary-color) !important;
        font-weight: 800;
    }}
    
    /* Progress Bar */
    .stProgress > div > div {{
        background: linear-gradient(90deg, var(--primary-color), var(--primary-light)) !important;
    }}
    
    /* Section Dividers */
    hr {{
        border-color: var(--border-color) !important;
        margin: 40px 0 !important;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        padding: 40px 0;
        color: var(--text-muted);
        font-size: 13px;
        border-top: 2px solid var(--border-color);
        margin-top: 60px;
    }}
    
    .footer strong {{
        color: var(--primary-color) !important;
    }}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .brand-title {{
            font-size: 24px;
        }}
        .metric-grid {{
            grid-template-columns: 1fr;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# PROFESSIONAL HEADER
# ==========================================
st.markdown(f"""
<div class="main-header">
    <div class="brand-container">
        {logo_html}
        <div class="brand-text">
            <div class="brand-title">PropPulse+</div>
            <div class="brand-subtitle">Advanced NBA Player Prop Analytics Platform</div>
        </div>
        <div class="status-badge">LIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR - MODE SELECTION & SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown("---")
    
    # Date Picker
    from datetime import datetime, timedelta
    
    # Default to "yesterday" if it's past midnight and no games today
    default_date = datetime.now()
    
    analysis_date = st.date_input(
        "Analysis Date",
        value=default_date,
        min_value=datetime.now() - timedelta(days=7),
        max_value=datetime.now() + timedelta(days=7),
        help="Select the date for opponent/schedule lookup. Use this if the current date shows no games."
    )
    
    # Show warning if selected date might have no games
    if analysis_date.strftime("%Y-%m-%d") != datetime.now().strftime("%Y-%m-%d"):
        st.caption(f"⚠️ Using custom date: {analysis_date.strftime('%b %d, %Y')}")
    
    st.markdown("---")
    
    mode = st.radio(
        "Select Analysis Mode",
        ["🎯 Single Prop Analysis", "📊 Batch Manual Entry", "📁 CSV Import"],
        index=0
    )
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        debug_mode = st.checkbox("Enable Debug Mode", value=True)
        show_charts = st.checkbox("Show Visualizations", value=True)
        confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    st.caption("**PropPulse+ v2025.5**")
    st.caption("L20-Weighted Projection Model")
    st.caption("Integrated FantasyPros DvP")
    st.caption("Built for Professional Bettors")

# ==========================================
# MODE 1: SINGLE PROP ANALYSIS
# ==========================================
if mode == "🎯 Single Prop Analysis":
    
    with st.form("prop_analyzer"):
        st.markdown("### 📋 Enter Prop Details")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            player = st.text_input(
                "Player Name",
                placeholder="LeBron James",
                help="Enter the full player name as it appears in official NBA stats"
            )
        
        with col2:
            stat = st.selectbox(
                "Stat Category",
                ["PTS", "REB", "AST", "REB+AST", "PRA", "P+R", "P+A", "FG3M"],
                help="Select the prop bet type"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            line = st.number_input(
                "Line",
                min_value=0.0,
                max_value=100.0,
                value=25.5,
                step=0.5,
                help="The over/under line from your sportsbook"
            )
        
        with col4:
            odds = st.number_input(
                "Odds (American)",
                value=-110,
                step=5,
                help="Enter odds in American format (e.g., -110, +150)"
            )
        
        st.markdown("---")
        submit = st.form_submit_button("🔍 ANALYZE PROP", use_container_width=True)

    if submit:
        if not player.strip():
            st.error("⚠️ Please enter a player name")
            st.stop()
        
        # Load settings
        try:
            settings = pe.load_settings()
        except Exception as e:
            st.error(f"❌ Failed to load settings: {e}")
            st.stop()
        
        # ==========================================
        # RUN ANALYSIS
        # ==========================================
        
        # Display current date being analyzed
        analysis_date_str = analysis_date.strftime("%Y-%m-%d")
        st.info(f"📅 Analyzing for date: **{analysis_date_str}** (Opponent matchup based on this date)")
        
        with st.spinner(f"🏀 Analyzing {player}'s {stat} projection..."):
            try:
                # Pass analysis date to settings
                settings['analysis_date'] = analysis_date_str
                
                buf = io.StringIO()
                with redirect_stdout(buf):
                    result = pe.analyze_single_prop(
                        player=player,
                        stat=stat,
                        line=line,
                        odds=int(odds),
                        settings=settings,
                        debug_mode=debug_mode
                    )

                model_output = buf.getvalue()

                if not result:
                    st.error("❌ Unable to analyze this prop. Player data may be unavailable.")
                    if debug_mode and model_output:
                        with st.expander("🔧 Debug Log"):
                            st.code(model_output)
                    st.stop()

            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
                import traceback
                with st.expander("Show Error Details"):
                    st.code(traceback.format_exc())
                st.stop()

        # ==========================================
        # RESULTS DISPLAY
        # ==========================================
        p_model = result['p_model']
        p_book = result['p_book']
        ev = result['ev']
        projection = result['projection']
        n_games = result['n_games']
        opponent = result.get('opponent', 'N/A')
        position = result.get('position', 'N/A')
        dvp_mult = result.get('dvp_mult', 1.0)
        confidence = result.get('confidence', 0.0)
        grade = result.get('grade', 'N/A')

        edge = (p_model - p_book) * 100
        ev_cents = ev * 100
        recommendation = "OVER" if projection > line else "UNDER"

        st.success("✅ Analysis Complete!")

        # Metric Cards
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Projection</div>
                <div class="metric-value">{projection:.1f}</div>
                <div class="metric-sublabel metric-neutral">{stat} • {n_games} games</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ev_class = "metric-positive" if ev_cents > 0 else "metric-negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Value</div>
                <div class="metric-value {ev_class}">{ev_cents:+.1f}¢</div>
                <div class="metric-sublabel metric-neutral">Per $1 wagered</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            edge_class = "metric-positive" if edge > 0 else "metric-negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Edge</div>
                <div class="metric-value {edge_class}">{edge:+.1f}%</div>
                <div class="metric-sublabel metric-neutral">vs. Sportsbook</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            conf_class = "metric-positive" if confidence > 0.7 else "metric-neutral"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value {conf_class}">{confidence:.2f}</div>
                <div class="metric-sublabel metric-neutral">{grade}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # EV Display
        ev_class = "ev-positive" if ev > 0 else "ev-negative"
        rec_text = f"BET {recommendation}" if ev > 0 else "FADE THIS PROP"
        
        st.markdown(f"""
        <div class="ev-container">
            <div class="ev-badge {ev_class}">
                {ev_cents:+.1f}¢ EV
            </div>
            <div class="recommendation">
                {rec_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Context Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <strong>📊 Model Analysis</strong><br>
                Model Probability: <strong>{p_model*100:.1f}%</strong><br>
                Sportsbook Implied: <strong>{p_book*100:.1f}%</strong><br>
                Line: <strong>{line}</strong><br>
                Odds: <strong>{odds}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <strong>🏀 Matchup Context</strong><br>
                Position: <strong>{position}</strong><br>
                Opponent: <strong>{opponent}</strong><br>
                DvP Multiplier: <strong>{dvp_mult:.3f}</strong><br>
                Sample Size: <strong>{n_games} games</strong>
            </div>
            """, unsafe_allow_html=True)

        # Visualizations
        if show_charts:
            st.markdown("---")
            st.markdown("### 📈 Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability Comparison
                fig = go.Figure(data=[
                    go.Bar(name='Model', x=['Model', 'Sportsbook'], y=[p_model*100, p_book*100],
                           marker_color=['#FF8C42', '#6B7280'])
                ])
                fig.update_layout(
                    title="Probability Comparison",
                    yaxis_title="Probability (%)",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # EV Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=ev_cents,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Expected Value (¢)"},
                    gauge={
                        'axis': {'range': [-20, 20]},
                        'bar': {'color': "#FF8C42"},
                        'steps': [
                            {'range': [-20, 0], 'color': "#374151"},
                            {'range': [0, 20], 'color': "#1F2937"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0
                        }
                    }
                ))
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Debug Output
        if debug_mode and model_output:
            with st.expander("🔧 Debug Log"):
                st.code(model_output)

# ==========================================
# MODE 2: BATCH MANUAL ENTRY
# ==========================================
elif mode == "📊 Batch Manual Entry":
    st.markdown("### 📊 Batch Manual Entry")

    st.info("""
    Enter multiple props manually for quick batch analysis.
    - Fill in player name, stat type, line, and odds
    - Click **Analyze Batch** to process all entries
    - Export results for further analysis
    """)

    n_props = st.number_input(
        "Number of props to analyze",
        min_value=1,
        max_value=20,
        value=3,
        help="How many props would you like to enter?"
    )

    manual_entries = []
    for i in range(int(n_props)):
        st.markdown(f"#### Prop #{i+1}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            player = st.text_input(f"Player", key=f"player_{i}", placeholder="Player Name")
        with col2:
            stat = st.selectbox(f"Stat", ["PTS", "REB", "AST", "REB+AST", "PRA", "P+R", "P+A", "FG3M"], key=f"stat_{i}")
        with col3:
            line = st.number_input(f"Line", min_value=0.0, max_value=100.0, value=20.0, key=f"line_{i}")
        with col4:
            odds = st.text_input(f"Odds", value="-110", key=f"odds_{i}")
        
        if player.strip():
            manual_entries.append({"player": player, "stat": stat, "line": line, "odds": odds})

    if manual_entries:
    df_preview = pd.DataFrame(manual_entries)
    st.subheader("📋 Preview")
    st.dataframe(df_preview, use_container_width=True)

    if st.button("🚀 ANALYZE BATCH", type="primary", use_container_width=True):
        if not manual_entries:
            st.error("⚠️ Please enter at least one valid player name.")
            st.stop()

        settings = pe.load_settings()
        settings['analysis_date'] = analysis_date.strftime("%Y-%m-%d")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        for i, entry in enumerate(manual_entries):
            status_text.text(f"Analyzing {i+1}/{len(manual_entries)}: {entry['player']}...")
            progress_bar.progress((i + 1) / len(manual_entries))
            
            try:
                line_val = float(entry["line"])
                odds_val = int(entry["odds"])
                result = pe.analyze_single_prop(entry["player"], entry["stat"], line_val, odds_val, settings)
                if result:
                    results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Skipped {entry['player']}: {str(e)}")

        progress_bar.empty()
        status_text.empty()

        if results:
            st.success(f"✅ Successfully analyzed {len(results)}/{len(manual_entries)} props!")
            results_df = pd.DataFrame(results)
            
            # Summary metrics
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            pos_ev = len([r for r in results if r['ev'] > 0])
            avg_ev = sum(r['ev'] for r in results) / len(results)
            top_ev = max(results, key=lambda x: x['ev'])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Positive EV Props</div>
                    <div class="metric-value">{pos_ev}</div>
                    <div class="metric-sublabel metric-neutral">Out of {len(results)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_class = "metric-positive" if avg_ev > 0 else "metric-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average EV</div>
                    <div class="metric-value {avg_class}">{avg_ev * 100:.1f}¢</div>
                    <div class="metric-sublabel metric-neutral">Per dollar</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Top EV Prop</div>
                    <div class="metric-value metric-positive">{top_ev['ev'] * 100:.1f}¢</div>
                    <div class="metric-sublabel metric-neutral">{top_ev['player']} {top_ev['stat']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Results Table
            st.markdown("---")
            st.subheader("📊 Detailed Results")
            
            display_df = results_df[[
                'player', 'stat', 'line', 'odds', 'projection', 'p_model', 'ev', 'confidence', 'grade'
            ]].copy()

            display_df['p_model'] = (display_df['p_model'] * 100).round(1)
            display_df['ev'] = (display_df['ev'] * 100).round(1)
            display_df['projection'] = display_df['projection'].round(1)
            display_df['confidence'] = display_df['confidence'].round(2)

            display_df.rename(columns={
                'player': 'Player',
                'stat': 'Stat',
                'line': 'Line',
                'odds': 'Odds',
                'projection': 'Projection',
                'p_model': 'Model %',
                'ev': 'EV (¢)',
                'confidence': 'Confidence',
                'grade': 'Grade'
            }, inplace=True)

            st.dataframe(
                display_df.sort_values('EV (¢)', ascending=False),
                use_container_width=True,
                height=400
            )

            # Visualization
            if show_charts:
                st.markdown("---")
                st.markdown("### 📈 Batch Analysis Visualization")
                
                fig = px.scatter(
                    display_df,
                    x='Confidence',
                    y='EV (¢)',
                    color='EV (¢)',
                    size='Projection',
                    hover_data=['Player', 'Stat', 'Line'],
                    title="EV vs Confidence",
                    color_continuous_scale=['#EF4444', '#6B7280', '#FF8C42', '#10B981']
                )
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

            # Export
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Download Results as CSV",
                csv_data,
                f"proppulse_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

        else:
            st.error("❌ No valid results generated. Please check your inputs and try again.")

# ==========================================
# MODE 3: CSV IMPORT
# ==========================================
else:
    st.markdown("### 📁 CSV Import")
    
    st.info("""
    Upload a CSV file with the following columns:
    - **player**: Player name
    - **stat**: Stat type (PTS, REB, AST, etc.)
    - **line**: Over/under line
    - **odds**: American odds format
    """)
    
    # Sample CSV download
    sample_data = pd.DataFrame({
        'player': ['LeBron James', 'Stephen Curry', 'Giannis Antetokounmpo'],
        'stat': ['PTS', 'FG3M', 'REB'],
        'line': [25.5, 3.5, 11.5],
        'odds': [-110, -120, -105]
    })
    
    sample_csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Sample CSV Template",
        sample_csv,
        "proppulse_template.csv",
        "text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['player', 'stat', 'line', 'odds']
            if not all(col in df_upload.columns for col in required_cols):
                st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
                st.stop()
            
            st.success(f"✅ Loaded {len(df_upload)} props from CSV")
            st.dataframe(df_upload, use_container_width=True)
            
            if st.button("🚀 ANALYZE CSV DATA", type="primary", use_container_width=True):
                settings = pe.load_settings()
                settings['analysis_date'] = analysis_date.strftime("%Y-%m-%d")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for idx, row in df_upload.iterrows():
                    status_text.text(f"Analyzing {idx+1}/{len(df_upload)}: {row['player']}...")
                    progress_bar.progress((idx + 1) / len(df_upload))
                    
                    try:
                        result = pe.analyze_single_prop(
                            player=str(row['player']),
                            stat=str(row['stat']),
                            line=float(row['line']),
                            odds=int(row['odds']),
                            settings=settings
                        )
                        if result:
                            results.append(result)
                    except Exception as e:
                        st.warning(f"⚠️ Skipped {row['player']}: {str(e)}")
                
                progress_bar.empty()
                status_text.empty()
                
                if results:
                    st.success(f"✅ Successfully analyzed {len(results)}/{len(df_upload)} props!")
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary metrics (same as batch mode)
                    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    
                    pos_ev = len([r for r in results if r['ev'] > 0])
                    avg_ev = sum(r['ev'] for r in results) / len(results)
                    top_ev = max(results, key=lambda x: x['ev'])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Positive EV Props</div>
                            <div class="metric-value">{pos_ev}</div>
                            <div class="metric-sublabel metric-neutral">Out of {len(results)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        avg_class = "metric-positive" if avg_ev > 0 else "metric-negative"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Average EV</div>
                            <div class="metric-value {avg_class}">{avg_ev * 100:.1f}¢</div>
                            <div class="metric-sublabel metric-neutral">Per dollar</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Top EV Prop</div>
                            <div class="metric-value metric-positive">{top_ev['ev'] * 100:.1f}¢</div>
                            <div class="metric-sublabel metric-neutral">{top_ev['player']} {top_ev['stat']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Results table
                    st.markdown("---")
                    st.subheader("📊 Detailed Results")
                    
                    display_df = results_df[[
                        'player', 'stat', 'line', 'odds', 'projection', 'p_model', 'ev', 'confidence', 'grade'
                    ]].copy()

                    display_df['p_model'] = (display_df['p_model'] * 100).round(1)
                    display_df['ev'] = (display_df['ev'] * 100).round(1)
                    display_df['projection'] = display_df['projection'].round(1)
                    display_df['confidence'] = display_df['confidence'].round(2)

                    display_df.rename(columns={
                        'player': 'Player',
                        'stat': 'Stat',
                        'line': 'Line',
                        'odds': 'Odds',
                        'projection': 'Projection',
                        'p_model': 'Model %',
                        'ev': 'EV (¢)',
                        'confidence': 'Confidence',
                        'grade': 'Grade'
                    }, inplace=True)

                    st.dataframe(
                        display_df.sort_values('EV (¢)', ascending=False),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Export
                    csv_data = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Download Results as CSV",
                        csv_data,
                        f"proppulse_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                else:
                    st.error("❌ No valid results generated.")
        
        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <strong>PropPulse+ v2025.5</strong> — Professional NBA Player Prop Analytics Platform<br>
    L20-Weighted Projection Model • Integrated FantasyPros DvP • Built for Professional Bettors<br>
    <br>
    <em>⚠️ For entertainment and educational purposes only. Bet responsibly.</em>
</div>
""", unsafe_allow_html=True)