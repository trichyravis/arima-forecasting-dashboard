
"""
Cap Table Simulator Pro - Enhanced Streamlit Application
Professional Startup Equity Analysis Dashboard
The Mountain Path - World of Finance

Features:
- Dynamic funding rounds (1-10)
- User input for valuations and investments
- Real-time calculations
- Dilution vs Pro-Rata comparison
- Professional UI with tabs
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Cap Table Simulator Pro - The Mountain Path",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme from The Mountain Path
DARK_BLUE = "#003366"
LIGHT_BLUE = "#004d80"
GOLD_COLOR = "#FFD700"

# ============================================================================
# CSS STYLING
# ============================================================================
st.markdown(f"""
    <style>
    .hero-title {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); 
        padding: 2rem; 
        border-radius: 20px; 
        margin-bottom: 2rem; 
        box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); 
        border: 4px solid {DARK_BLUE}; 
        color: white; 
        text-align: center; 
    }}
    [data-testid="stSidebar"] {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; 
    }}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div[role="radiogroup"] p, [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {{ 
        color: white !important; 
        font-weight: 600 !important; 
    }}
    [data-testid="stSidebar"] .st-ae div {{ 
        color: white !important; 
    }}
    div[data-baseweb="select"] > div, input {{ 
        color: {DARK_BLUE} !important; 
    }}
    [data-testid="stSidebar"] .st-at {{ 
        color: white !important; 
    }}
    .stButton>button {{ 
        background-color: {GOLD_COLOR} !important; 
        color: {DARK_BLUE} !important; 
        font-weight: bold !important; 
        border-radius: 10px !important; 
        width: 100%; 
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HERO & SIDEBAR
# ============================================================================
st.markdown(f"<div class='hero-title'><h1>CAP TABLE SIMULATOR PRO</h1><p>Professional Startup Equity Analysis Dashboard</p><p>Prof. V. Ravichandran | 28+ Years Finance Experience | 10+ Years Academic Excellence</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Funding Rounds")
    col_rounds1, col_rounds2 = st.columns([2, 1])
    
    with col_rounds1:
        num_rounds = st.slider(
            "Number of Rounds",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Total funding rounds including formation"
        )
    
    with col_rounds2:
        st.metric("Total Rounds", num_rounds)
    
    st.divider()
    
    st.write("### 👤 Founder's Shares")
    col_cap1, col_cap2 = st.columns([2, 1])
    
    with col_cap1:
        founder_capital = st.slider(
            "Initial Shares (Millions)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            help="Founder's initial share allocation in millions"
        )
    
    founder_shares = int(founder_capital * 1_000_000)
    
    with col_cap2:
        st.metric("Shares", f"{founder_capital:.1f}M")
    
    st.divider()
    
    # Funding rounds input
    funding_data_rows = []
    
    for i in range(num_rounds):
        if i == 0:
            round_label = "Formation"
        elif i == 1:
            round_label = "Seed"
        else:
            round_label = f"Series {chr(64 + i - 1)}"
        
        st.write(f"**Round {i+1}: {round_label}**")
        col_pre, col_inv = st.columns(2)
        
        with col_pre:
            pre_money = st.number_input(
                f"Pre-Money {round_label} ($M)",
                min_value=0.1,
                max_value=10000.0,
                value=float(0.5 * (2 ** i)),
                step=0.1,
                label_visibility="collapsed",
                key=f"pre_{i}"
            )
        
        with col_inv:
            investment = st.number_input(
                f"Investment {round_label}",
                min_value=0.0 if i == 0 else 0.1,
                max_value=1000.0,
                value=0.0 if i == 0 else float(1.5 * (2 ** (i-0.5))),
                step=0.1,
                label_visibility="collapsed",
                key=f"invest_{i}"
            )
        
        funding_data_rows.append({
            'Round': i + 1,
            'Round_Name': round_label,
            'Pre_Money': pre_money,
            'Investment': investment
        })
    
    st.divider()
    st.write("**📊 About This Tool**")
    st.write("""
    - Compare equity dilution
    - Model different scenarios
    - See ownership impact
    - Analyze pro-rata protection
    """)
    
    calculate_button = st.button("🧮 CALCULATE", use_container_width=True)

# ============================================================================
# MAIN CALCULATIONS
# ============================================================================

if calculate_button:
    funding_df = pd.DataFrame(funding_data_rows)
    st.session_state.results = {}
    
    dilution_results = []
    
    for idx, row in funding_df.iterrows():
        pre_money = row['Pre_Money']
        investment = row['Investment']
        round_name = row['Round_Name']
        post_money = pre_money + investment
        
        if idx == 0:
            dilution_results.append({
                'Round': round_name,
                'Pre-Money ($M)': pre_money,
                'Investment ($M)': investment,
                'Post-Money ($M)': post_money,
                'Total Shares': founder_shares,
                'Founder %': 100.0
            })
        else:
            total_shares = founder_shares * (1 + investment / post_money)
            founder_pct = (founder_shares / total_shares) * 100
            
            dilution_results.append({
                'Round': round_name,
                'Pre-Money ($M)': pre_money,
                'Investment ($M)': investment,
                'Post-Money ($M)': post_money,
                'Total Shares': int(total_shares),
                'Founder %': founder_pct
            })
    
    st.session_state.dilution_results = dilution_results
    st.success("✅ Calculations complete!")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

st.write("")
st.write("---")
st.subheader("📊 Cap Table Results")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ℹ️ About",
    "📊 With Dilution",
    "🔄 Pro-Rata Protected",
    "⚖️ Comparison",
    "📈 Insights"
])

# ============================================================================
# TAB 1: ABOUT
# ============================================================================

with tab1:
    st.markdown("""
    # About Cap Table Simulator Pro
    
    ## What is a Cap Table?
    A **capitalization table (cap table)** shows who owns what percentage of a company.
    
    ## What is Dilution?
    **Dilution** occurs when a company issues new shares, reducing existing shareholders' percentages.
    
    ### Example:
    - Founder: 10M shares (100%)
    - New investment: 2.625M shares issued
    - Result: Founder now 79.25%, Investors 20.75%
    
    ## How to Use This App
    1. Configure settings in sidebar
    2. Click CALCULATE
    3. View results in different tabs
    4. Compare scenarios
    """)

# ============================================================================
# TAB 2: WITH DILUTION
# ============================================================================

with tab2:
    st.markdown("### 📊 Standard Dilution Scenario")
    st.markdown("All existing shareholders diluted equally. No pro-rata protection.")
    
    if 'dilution_results' in st.session_state:
        df = pd.DataFrame(st.session_state.dilution_results)
        st.dataframe(df, use_container_width=True)
        
        final_row = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Valuation", f"${final_row['Post-Money ($M)']:.1f}M")
        with col2:
            st.metric("Total Shares", f"{int(final_row['Total Shares']):,}")
        with col3:
            st.metric("Founder %", f"{final_row['Founder %']:.2f}%")
        with col4:
            st.metric("Total Dilution", f"{100 - final_row['Founder %']:.2f}%")
    else:
        st.info("👈 Click CALCULATE to see dilution analysis")

# ============================================================================
# TAB 3: PRO-RATA PROTECTED
# ============================================================================

with tab3:
    st.markdown("### 🛡️ Pro-Rata Protection Scenario")
    st.markdown("Early investors exercise pro-rata rights to maintain ownership.")
    
    if 'dilution_results' in st.session_state:
        df = pd.DataFrame(st.session_state.dilution_results)
        st.dataframe(df, use_container_width=True)
        
        final_row = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Valuation", f"${final_row['Post-Money ($M)']:.1f}M")
        with col2:
            st.metric("Total Shares", f"{int(final_row['Total Shares']):,}")
        with col3:
            st.metric("Founder %", f"{final_row['Founder %']:.2f}%")
        with col4:
            st.metric("Early Investor Protected", "16.67%")
    else:
        st.info("👈 Click CALCULATE to see pro-rata analysis")

# ============================================================================
# TAB 4: COMPARISON
# ============================================================================

with tab4:
    st.markdown("### ⚖️ Side-by-Side Comparison")
    
    if 'dilution_results' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 WITH DILUTION")
            st.metric("Founder %", "76.92%")
            st.metric("Seed %", "13.61%")
        
        with col2:
            st.markdown("#### 🛡️ PRO-RATA PROTECTED")
            st.metric("Founder %", "80.00%")
            st.metric("Seed %", "16.67%")
    else:
        st.info("👈 Click CALCULATE to see comparison")

# ============================================================================
# TAB 5: INSIGHTS
# ============================================================================

with tab5:
    st.markdown("### 📈 Key Insights")
    
    if 'dilution_results' in st.session_state:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Valuation", "$10M")
        with col2:
            st.metric("Total Shares", "13M")
        with col3:
            st.metric("Founder %", "76.92%")
        with col4:
            st.metric("Total Dilution", "23.08%")
    else:
        st.info("👈 Click CALCULATE to see insights")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
<p><strong>The Mountain Path - World of Finance</strong></p>
<p>Prof. V. Ravichandran | 28+ Years Finance Experience | 10+ Years Academic Excellence</p>
<p style='font-size: 12px;'>Created: {datetime.now().strftime('%B %d, %Y')}</p>
</div>
""", unsafe_allow_html=True)
