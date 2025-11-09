"""
PropPulse+ v2025.3 — Professional NBA Prop Analyzer
Modern betting interface with dark theme
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
# CUSTOM CSS - Modern Betting UI
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    
    .logo-text {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .tagline {
        color: #8B92B0;
        font-size: 14px;
        margin-top: 0.5rem;
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
    }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 8px;
    }
    
    .metric-label {
        color: #8B92B0;
        font-size: 12px;
        font-weight: 600;
    }
    
    .metric-value {
        color: white;
        font-size: 24px;
        font-weight: 700;
        margin: 8px 0;
    }
    
    .ev-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 700;
        display: inline-block;
    }
    
    .ev-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR - MODE SELECTION
# ==========================================
with st.sidebar:
    st.markdown("### 🏀 PropPulse+")
    st.markdown("---")
    
   mode = st.radio(
    "Analysis Mode",
    ["🎯 Single Prop", "📊 Batch Manual Entry"],
    index=0
)

    
    st.markdown("---")
    
    if mode == "🎯 Single Prop":
        debug_mode = st.checkbox("🔧 Debug Mode", value=False)
    else:
        debug_mode = False
    
    st.markdown("---")
    st.caption("PropPulse+ v2025.3")
    st.caption("L20-Weighted Model")

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="logo-text">PropPulse+</div>
    <div class="tagline">AI-Powered NBA Player Prop Analytics</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MODE 1: SINGLE PROP
# ==========================================
if mode == "🎯 Single Prop":
    
    with st.form("prop_analyzer"):
        player = st.text_input(
            "Player Name",
            placeholder="e.g., LeBron James",
            help="Enter the full player name"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            stat = st.selectbox(
                "Stat Category",
                ["PTS", "REB", "AST", "REB+AST", "PRA", "FG3M"],
                help="Select the prop type"
            )
        
        with col2:
            line = st.number_input(
                "Line",
                min_value=0.0,
                max_value=100.0,
                value=25.5,
                step=0.5,
                help="The sportsbook line"
            )
        
        col3, col4 = st.columns([2, 1])
        with col3:
            odds = st.number_input(
                "Odds (American)",
                value=-110,
                step=5,
                help="Enter American odds (e.g., -110 or +150)"
            )
        
        with col4:
            st.write("")
            st.write("")
            submit = st.form_submit_button("🔍 Analyze Prop")

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
                
                # Calculate metrics
                edge = (p_model - p_book) * 100
                ev_cents = ev * 100
                recommendation = "OVER" if projection > line else "UNDER"
                
                # ==========================================
                # RESULTS DISPLAY
                # ==========================================
                
                st.success("✅ Analysis Complete!")
                
                # Main metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">PROJECTION</div>
                        <div class="metric-value">{projection:.1f}</div>
                        <div style="color: {'#11998e' if projection > line else '#eb3349'}; font-size: 13px;">
                            {projection - line:+.1f} vs line
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">MODEL PROB</div>
                        <div class="metric-value">{p_model * 100:.1f}%</div>
                        <div style="color: {'#11998e' if edge > 0 else '#eb3349'}; font-size: 13px;">
                            {edge:+.1f}% edge
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">BOOK PROB</div>
                        <div class="metric-value">{p_book * 100:.1f}%</div>
                        <div style="color: #8B92B0; font-size: 13px;">
                            Implied
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">GAMES</div>
                        <div class="metric-value">{n_games}</div>
                        <div style="color: #8B92B0; font-size: 13px;">
                            Analyzed
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # EV Display
                ev_class = "ev-positive" if ev_cents > 0 else "ev-negative"
                ev_emoji = "📈" if ev_cents > 0 else "📉"
                st.markdown(f"""
                <div style="text-align: center; margin: 24px 0;">
                    <div class="{ev_class}">
                        {ev_emoji} EV: {ev_cents:+.1f}¢ per $1
                    </div>
                    <div style="color: white; margin-top: 16px; font-size: 18px; font-weight: 600;">
                        Recommendation: {recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Context Info
                st.markdown(f"""
                <div class="result-card">
                    <strong>Context:</strong> {position} vs {opponent} • DvP: {dvp_mult:.3f}
                </div>
                """, unsafe_allow_html=True)
                
                # Debug output
                if debug_mode and model_output:
                    with st.expander("🔧 Model Debug Log", expanded=True):
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
    st.markdown("### 🧠 Batch Manual Entry")

    st.info("""
    Enter multiple props manually for quick batch analysis without a CSV file.
    - Fill in player name, stat type, line, and odds
    - Click **Analyze Batch** to process all at once
    """)

    # Choose how many props to enter
    n_props = st.number_input(
        "How many props would you like to analyze?",
        min_value=1,
        max_value=20,
        value=3
    )

    manual_entries = []
    for i in range(int(n_props)):
        st.markdown(f"#### Prop #{i+1}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            player = st.text_input(f"Player {i+1}", key=f"player_{i}", placeholder="e.g., LeBron James")
        with col2:
            stat = st.selectbox(
                f"Stat {i+1}",
                ["PTS", "REB", "AST", "REB+AST", "PRA", "FG3M"],
                key=f"stat_{i}"
            )
        with col3:
            line = st.number_input(f"Line {i+1}", min_value=0.0, max_value=100.0, value=20.0, key=f"line_{i}")
        with col4:
            odds = st.text_input(f"Odds {i+1}", value="-110", key=f"odds_{i}")
        
        if player.strip():
            manual_entries.append({
                "player": player,
                "stat": stat,
                "line": line,
                "odds": odds
            })

    # Optional: Save these manual entries as a CSV for reuse
    if manual_entries:
        df_preview = pd.DataFrame(manual_entries)
        st.subheader("📋 Preview")
        st.dataframe(df_preview, use_container_width=True)
        csv_data = df_preview.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download These Entries as CSV", csv_data, "manual_entries.csv")

    # Analyze Button
    if st.button("🚀 Analyze Batch", type="primary"):
        if not manual_entries:
            st.error("⚠️ Please enter at least one valid player name.")
            st.stop()

        settings = pe.load_settings()
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        for i, entry in enumerate(manual_entries):
            status_text.text(f"Analyzing {i+1}/{len(manual_entries)}: {entry['player']}...")
            progress_bar.progress((i + 1) / len(manual_entries))
            try:
                result = pe.analyze_single_prop(
                    entry["player"],
                    entry["stat"],
                    entry["line"],
                    entry["odds"],
                    settings
                )
                if result:
                    results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Skipped {entry['player']}: {str(e)}")

        progress_bar.empty()
        status_text.empty()

        if results:
            st.success(f"✅ Successfully analyzed {len(results)}/{len(manual_entries)} props!")

            results_df = pd.DataFrame(results)

            # Summary Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                pos_ev = len([r for r in results if r['ev'] > 0])
                st.metric("Positive EV Props", pos_ev)
            with col2:
                avg_ev = sum(r['ev'] for r in results) / len(results)
                st.metric("Average EV", f"{avg_ev * 100:.1f}¢")
            with col3:
                top_ev = max(results, key=lambda x: x['ev'])
                st.metric("Top EV", f"{top_ev['ev'] * 100:.1f}¢")

            # Results Table
            st.subheader("📊 All Results")
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

            # Export button
            if st.button("💾 Generate Excel Dashboard"):
                with st.spinner("Generating Excel..."):
                    pe.export_results_to_excel(results_df)
                    st.success("✅ Excel file generated in /output folder!")
        else:
            st.error("❌ No valid results generated. Double-check your inputs.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8B92B0; padding: 2rem 0; font-size: 13px;">
    PropPulse+ v2025.3 | L20-Weighted Model with FantasyPros DvP<br>
    Built with ❤️ for sharper betting decisions
</div>
""", unsafe_allow_html=True)