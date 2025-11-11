"""
PropPulse+ v2025.4 — Professional NBA Prop Analyzer
Industrial UI with enhanced accessibility and new bet types
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import io
from contextlib import redirect_stdout

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
# CUSTOM CSS - Professional Industrial UI
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Force dark theme colors */
    .stApp {
        background: #0B0F19;
        font-family: 'Inter', sans-serif;
    }
    
    /* Text contrast fix for light mode users */
    .stApp, .stMarkdown, p, span, label, div {
        color: #E5E7EB !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #F9FAFB !important;
        font-weight: 700 !important;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        border-bottom: 1px solid #374151;
        padding: 1.5rem 2rem;
        margin: -6rem -2rem 2rem -2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .brand-container {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .brand-logo {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: 800;
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    .brand-text {
        flex: 1;
    }
    
    .brand-title {
        font-size: 28px;
        font-weight: 800;
        color: #F9FAFB;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1;
    }
    
    .brand-subtitle {
        font-size: 13px;
        color: #9CA3AF;
        margin-top: 4px;
        font-weight: 500;
    }
    
    .status-badge {
        background: #065F46;
        color: #10B981;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #10B981;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #374151;
    }
    
    [data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }
    
    /* Form Controls */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: #1F2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        color: #F9FAFB !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Labels */
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label {
        color: #D1D5DB !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 8px !important;
    }
    
    /* Primary Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 16px 24px;
        font-weight: 700;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 24px 0;
    }
    
    .metric-card {
        background: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #4B5563;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .metric-label {
        color: #9CA3AF;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #F9FAFB;
        font-size: 32px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 8px;
    }
    
    .metric-sublabel {
        font-size: 13px;
        font-weight: 500;
        margin-top: 8px;
    }
    
    .metric-positive {
        color: #10B981;
    }
    
    .metric-negative {
        color: #EF4444;
    }
    
    .metric-neutral {
        color: #6B7280;
    }
    
    /* EV Display */
    .ev-container {
        text-align: center;
        padding: 32px;
        background: #1F2937;
        border: 2px solid #374151;
        border-radius: 16px;
        margin: 24px 0;
    }
    
    .ev-badge {
        display: inline-block;
        padding: 16px 32px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .ev-positive {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        border: 2px solid #10B981;
    }
    
    .ev-negative {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        border: 2px solid #EF4444;
    }
    
    .recommendation {
        color: #F9FAFB;
        font-size: 20px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info Cards */
    .info-card {
        background: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    }
    
    .info-card strong {
        color: #F9FAFB !important;
        font-weight: 700;
    }
    
    /* Data Tables */
    [data-testid="stDataFrame"] {
        background: #1F2937;
        border-radius: 12px;
        border: 1px solid #374151;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: #064E3B !important;
        border: 1px solid #10B981 !important;
        color: #D1FAE5 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: #7F1D1D !important;
        border: 1px solid #EF4444 !important;
        color: #FEE2E2 !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background: #78350F !important;
        border: 1px solid #F59E0B !important;
        color: #FEF3C7 !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background: #1E3A8A !important;
        border: 1px solid #3B82F6 !important;
        color: #DBEAFE !important;
        border-radius: 8px !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: #3B82F6 !important;
    }
    
    /* Radio Buttons */
    [data-testid="stRadio"] label {
        color: #E5E7EB !important;
        font-weight: 500 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #E5E7EB !important;
        font-weight: 500 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1F2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        color: #F9FAFB !important;
        font-weight: 600 !important;
    }
    
    /* Section Dividers */
    hr {
        border-color: #374151 !important;
        margin: 32px 0 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 32px 0;
        color: #6B7280;
        font-size: 13px;
        border-top: 1px solid #374151;
        margin-top: 48px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PROFESSIONAL HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="brand-container">
        <div class="brand-logo">PP</div>
        <div class="brand-text">
            <div class="brand-title">PropPulse+</div>
            <div class="brand-subtitle">Advanced NBA Player Prop Analytics</div>
        </div>
        <div class="status-badge">● LIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR - MODE SELECTION
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown("---")
    
    mode = st.radio(
        "Select Mode",
        ["🎯 Single Prop Analysis", "📊 Batch Manual Entry"],
        index=0
    )
    
    st.markdown("---")
    
    if mode == "🎯 Single Prop Analysis":
        debug_mode = st.checkbox("🔧 Enable Debug Mode", value=True)
    else:
        debug_mode = False
    
    st.markdown("---")
    st.caption("**PropPulse+ v2025.4**")
    st.caption("L20-Weighted Projection Model")
    st.caption("FantasyPros DvP Integration")

# ==========================================
# MODE 1: SINGLE PROP ANALYSIS
# ==========================================
if mode == "🎯 Single Prop Analysis":
    
    with st.form("prop_analyzer"):
        st.markdown("### 📋 Prop Details")
        
        player = st.text_input(
            "Player Name",
            placeholder="LeBron James",
            help="Enter the full player name as it appears in official NBA stats"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            stat = st.selectbox(
                "Stat Category",
                ["PTS", "REB", "AST", "REB+AST", "PRA", "P+R", "P+A", "FG3M"],
                help="Select the prop bet type"
            )
        
        with col2:
            line = st.number_input(
                "Line",
                min_value=0.0,
                max_value=100.0,
                value=25.5,
                step=0.5,
                help="The over/under line from your sportsbook"
            )
        
        odds = st.number_input(
            "Odds (American Format)",
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
        
        with st.spinner(f"🏀 Analyzing {player}'s {stat} projection..."):
            try:
                # Capture model output
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
                
                # Extract results
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
                
                # Calculate metrics
                edge = (p_model - p_book) * 100
                ev_cents = ev * 100
                recommendation = "OVER" if projection > line else "UNDER"
                projection_diff = projection - line
                
                # ==========================================
                # RESULTS DISPLAY
                # ==========================================
                
                st.success("✅ Analysis Complete!")
                
                # Main metrics grid
                st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    diff_class = "metric-positive" if projection_diff > 0 else "metric-negative"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Projection</div>
                        <div class="metric-value">{projection:.1f}</div>
                        <div class="metric-sublabel {diff_class}">
                            {projection_diff:+.1f} vs line
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    edge_class = "metric-positive" if edge > 0 else "metric-negative"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Model Probability</div>
                        <div class="metric-value">{p_model * 100:.1f}%</div>
                        <div class="metric-sublabel {edge_class}">
                            {edge:+.1f}% edge
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Book Probability</div>
                        <div class="metric-value">{p_book * 100:.1f}%</div>
                        <div class="metric-sublabel metric-neutral">
                            Implied odds
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Sample Size</div>
                        <div class="metric-value">{n_games}</div>
                        <div class="metric-sublabel metric-neutral">
                            Games analyzed
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # EV Display
                ev_class = "ev-positive" if ev_cents > 0 else "ev-negative"
                ev_emoji = "📈" if ev_cents > 0 else "📉"
                st.markdown(f"""
                <div class="ev-container">
                    <div class="ev-badge {ev_class}">
                        {ev_emoji} EV: {ev_cents:+.1f}¢ per $1
                    </div>
                    <div class="recommendation">
                        Lean: {recommendation}
                    </div>
                    <div style="margin-top: 12px; color: #9CA3AF; font-size: 14px;">
                        Grade: {grade} • Confidence: {confidence:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Context Info
                st.markdown(f"""
                <div class="info-card">
                    <strong>Matchup Context:</strong> {position} vs {opponent} • 
                    <strong>DvP Multiplier:</strong> {dvp_mult:.3f}x • 
                    <strong>Games Analyzed:</strong> {n_games}
                </div>
                """, unsafe_allow_html=True)
                
                # Debug output
                if debug_mode and model_output:
                    with st.expander("🔧 Model Debug Log", expanded=False):
                        st.code(model_output, language="text")
            
            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
                import traceback
                error_details = traceback.format_exc()
                with st.expander("Show Error Details"):
                    st.code(error_details)
                
                # Try to show any captured output
                if 'model_output' in locals() and model_output:
                    with st.expander("Partial Model Output"):
                        st.code(model_output)

# ==========================================
# MODE 2: BATCH MANUAL ENTRY
# ==========================================
else:
    st.markdown("### 📊 Batch Manual Entry")

    st.info("""
    Enter multiple props manually for quick batch analysis.
    - Fill in player name, stat type, line, and odds
    - Click **Analyze Batch** to process all entries
    - Export results to Excel for further analysis
    """)

    # Choose how many props to enter
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
            player = st.text_input(
                f"Player",
                key=f"player_{i}",
                placeholder="LeBron James"
            )
        with col2:
            stat = st.selectbox(
                f"Stat",
                ["PTS", "REB", "AST", "REB+AST", "PRA", "P+R", "P+A", "FG3M"],
                key=f"stat_{i}"
            )
        with col3:
            line = st.number_input(
                f"Line",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                key=f"line_{i}"
            )
        with col4:
            odds = st.text_input(
                f"Odds",
                value="-110",
                key=f"odds_{i}"
            )
        
        if player.strip():
            manual_entries.append({
                "player": player,
                "stat": stat,
                "line": line,
                "odds": odds
            })

    # Preview table
    if manual_entries:
        df_preview = pd.DataFrame(manual_entries)
        st.subheader("📋 Preview")
        st.dataframe(df_preview, use_container_width=True)
        
        csv_data = df_preview.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download Entries as CSV",
            csv_data,
            "manual_entries.csv",
            use_container_width=True
        )

    # Analyze Button
    if st.button("🚀 ANALYZE BATCH", type="primary", use_container_width=True):
        if not manual_entries:
            st.error("⚠️ Please enter at least one valid player name.")
            st.stop()

        settings = pe.load_settings()
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        # Process each entry
        for i, entry in enumerate(manual_entries):
            status_text.text(f"Analyzing {i+1}/{len(manual_entries)}: {entry['player']}...")
            progress_bar.progress((i + 1) / len(manual_entries))
            
            try:
                # Convert inputs
                try:
                    line_val = float(entry["line"])
                except ValueError:
                    line_val = 0.0

                try:
                    odds_val = int(entry["odds"])
                except ValueError:
                    odds_val = -110

                # Run analysis
                result = pe.analyze_single_prop(
                    entry["player"],
                    entry["stat"],
                    line_val,
                    odds_val,
                    settings
                )

                if result:
                    results.append(result)

            except Exception as e:
                st.warning(f"⚠️ Skipped {entry['player']}: {str(e)}")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        if results:
            st.success(f"✅ Successfully analyzed {len(results)}/{len(manual_entries)} props!")

            results_df = pd.DataFrame(results)

            # Summary Metrics
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pos_ev = len([r for r in results if r['ev'] > 0])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Positive EV Props</div>
                    <div class="metric-value">{pos_ev}</div>
                    <div class="metric-sublabel metric-neutral">
                        Out of {len(results)} total
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_ev = sum(r['ev'] for r in results) / len(results)
                avg_class = "metric-positive" if avg_ev > 0 else "metric-negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average EV</div>
                    <div class="metric-value {avg_class}">{avg_ev * 100:.1f}¢</div>
                    <div class="metric-sublabel metric-neutral">
                        Per dollar wagered
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                top_ev = max(results, key=lambda x: x['ev'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Top EV Prop</div>
                    <div class="metric-value metric-positive">{top_ev['ev'] * 100:.1f}¢</div>
                    <div class="metric-sublabel metric-neutral">
                        {top_ev['player']} {top_ev['stat']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Results Table
            st.subheader("📊 Detailed Results")
            display_df = results_df[[
                'player', 'stat', 'line', 'odds', 'projection', 'p_model', 'ev', 'n_games'
            ]].copy()

            display_df['p_model'] = (display_df['p_model'] * 100).round(1)
            display_df['ev'] = (display_df['ev'] * 100).round(1)
            display_df['projection'] = display_df['projection'].round(1)

            display_df.rename(columns={
                'player': 'Player',
                'stat': 'Stat',
                'line': 'Line',
                'odds': 'Odds',
                'projection': 'Proj',
                'p_model': 'Model%',
                'ev': 'EV(¢)',
                'n_games': 'Games'
            }, inplace=True)

            st.dataframe(
                display_df.sort_values('EV(¢)', ascending=False),
                use_container_width=True,
                height=400
            )

            # Export functionality
            if st.button("💾 GENERATE EXCEL DASHBOARD", use_container_width=True):
                with st.spinner("Generating Excel dashboard..."):
                    try:
                        pe.export_results_to_excel(results_df)
                        st.success("✅ Excel dashboard generated in /output folder!")
                    except Exception as e:
                        st.error(f"❌ Export failed: {e}")

        else:
            st.error("❌ No valid results generated. Please check your inputs and try again.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <strong>PropPulse+ v2025.4</strong> | L20-Weighted Projection Model<br>
    Integrated with FantasyPros DvP • Built for Professional Bettors<br>
    <em>For entertainment purposes only. Bet responsibly.</em>
</div>
""", unsafe_allow_html=True)