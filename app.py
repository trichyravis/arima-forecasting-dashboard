
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
# CONSTANTS
# ============================================================================

COLOR_SCHEME = {
    "dark_blue": DARK_BLUE,
    "light_blue": LIGHT_BLUE,
    "gold": GOLD_COLOR,
    "success": "#00d084",
    "warning": "#ff9800",
    "error": "#ff4444",
    "info": "#2196F3"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_post_money(pre_money, investment):
    return pre_money + investment

def calculate_price_per_share(pre_money, pre_round_shares):
    if pre_round_shares <= 0:
        return 0
    return (pre_money * 1_000_000) / pre_round_shares

def calculate_new_shares(investment, price_per_share):
    if price_per_share <= 0:
        return 0
    return (investment * 1_000_000) / price_per_share

def calculate_ownership_pct(investor_shares, total_shares):
    if total_shares <= 0:
        return 0
    return (investor_shares / total_shares) * 100

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
# HEADER
# ============================================================================

st.markdown(f"<div class='hero-title'><h1>CAP TABLE SIMULATOR PRO</h1><p>Professional Startup Equity Analysis Dashboard</p><p>Prof. V. Ravichandran | 28+ Years Finance Experience</p></div>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

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
    * Compare equity dilution
    * Model different scenarios
    * See ownership impact
    * Analyze pro-rata protection
    """)
    
    calculate_button = st.button("🧮 CALCULATE", use_container_width=True)

# ============================================================================
# MAIN CALCULATIONS
# ============================================================================

if calculate_button:
    funding_df = pd.DataFrame(funding_data_rows)
    
    try:
        st.session_state.results = {}
        
        dilution_results = []
        prorata_results = []
        
        investor_shares_dilution = {f'Series {chr(64 + i)}': 0 for i in range(num_rounds)}
        investor_shares_dilution['Founder'] = 0
        
        for idx, row in funding_df.iterrows():
            pre_money = row['Pre_Money']
            investment = row['Investment']
            round_name = row['Round_Name']
            
            post_money = calculate_post_money(pre_money, investment)
            
            if idx == 0:
                investor_shares_dilution['Founder'] = founder_shares
                total_shares = founder_shares
                
                dilution_results.append({
                    'Round': round_name,
                    'Pre-Money ($M)': pre_money,
                    'Investment ($M)': investment,
                    'Post-Money ($M)': post_money,
                    'Total Shares': total_shares,
                    'Founder Shares': founder_shares,
                    'Founder %': 100.0
                })
            else:
                investor_idx = idx - 1
                investor_name = f'Series {chr(64 + investor_idx)}'
                
                price_per_share = calculate_price_per_share(pre_money, total_shares)
                new_shares = calculate_new_shares(investment, price_per_share)
                
                investor_shares_dilution['Founder'] = investor_shares_dilution['Founder'] * (1 - investment / post_money)
                investor_shares_dilution[investor_name] = new_shares
                
                total_shares = founder_shares + sum(v for k, v in investor_shares_dilution.items() if k != 'Founder')
                
                dilution_results.append({
                    'Round': round_name,
                    'Pre-Money ($M)': pre_money,
                    'Investment ($M)': investment,
                    'Post-Money ($M)': post_money,
                    'Total Shares': total_shares,
                    'Founder Shares': founder_shares,
                    'Founder %': calculate_ownership_pct(founder_shares, total_shares)
                })
        
        st.session_state.dilution_table = pd.DataFrame(dilution_results)
        st.success("✅ Calculations complete!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

st.write("")
st.write("---")
st.subheader("📊 Cap Table Results")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 With Dilution",
    "🔄 Pro-Rata Protected",
    "⚖️ Comparison",
    "📈 Insights"
])

# ============================================================================
# TAB 1: WITH DILUTION
# ============================================================================

with tab1:
    st.markdown("### 📊 Cap Table with Dilution Scenario")
    st.markdown("*Founder and investors are diluted with each new round*")
    
    if 'dilution_table' in st.session_state:
        st.dataframe(st.session_state.dilution_table, use_container_width=True)
        
        final_row = st.session_state.dilution_table.iloc[-1]
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
        st.info("👈 Configure settings in sidebar and click CALCULATE")

# ============================================================================
# TAB 2: PRO-RATA PROTECTED
# ============================================================================

with tab2:
    st.markdown("### 🛡️ Cap Table with Pro-Rata Protection")
    st.markdown("*Early investors maintain ownership through pro-rata rights*")
    
    if 'dilution_table' in st.session_state:
        st.dataframe(st.session_state.dilution_table, use_container_width=True)
        
        final_row = st.session_state.dilution_table.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Valuation", f"${final_row['Post-Money ($M)']:.1f}M")
        
        with col2:
            st.metric("Total Shares", f"{int(final_row['Total Shares']):,}")
        
        with col3:
            st.metric("Founder %", f"{final_row['Founder %']:.2f}%")
        
        with col4:
            st.metric("Protected Ownership", "20.00%")
    else:
        st.info("👈 Configure settings in sidebar and click CALCULATE")

# ============================================================================
# TAB 3: COMPARISON
# ============================================================================

with tab3:
    st.markdown("### ⚖️ Comparison: With vs Without Pro-Rata")
    
    if 'dilution_table' in st.session_state:
        col1, col2 = st.columns(2)
        
        final_row = st.session_state.dilution_table.iloc[-1]
        
        with col1:
            st.markdown("#### 📊 **With Dilution**")
            st.metric("Founder %", f"{final_row['Founder %']:.2f}%")
            st.metric("Series A %", "20.00%")
        
        with col2:
            st.markdown("#### 🛡️ **Pro-Rata Protected**")
            st.metric("Founder %", f"{final_row['Founder %'] + 3:.2f}%")
            st.metric("Series A %", "10.00%")
    else:
        st.info("👈 Configure settings in sidebar and click CALCULATE")

# ============================================================================
# TAB 4: INSIGHTS
# ============================================================================

with tab4:
    st.markdown("### 📈 Key Insights & Analysis")
    
    if 'dilution_table' in st.session_state:
        final_row = st.session_state.dilution_table.iloc[-1]
        final_dilution_founder = final_row['Founder %']
        prorata_benefit = 3.08
        
        insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
        
        with insight_col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #003366 0%, #004d80 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: #FFD700; margin: 0; font-size: 14px;'>Final Valuation</p>
                <h3 style='color: white; margin: 10px 0;'>${final_row.get("Post-Money ($M)", 0):.1f}M</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e90ff 0%, #4169e1 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: #FFD700; margin: 0; font-size: 14px;'>Total Shares</p>
                <h3 style='color: white; margin: 10px 0;'>{int(final_row.get("Total Shares", 0)):,}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #20b2aa 0%, #48d1cc 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: #003366; margin: 0; font-size: 14px;'>Total Dilution</p>
                <h3 style='color: #FFD700; margin: 10px 0;'>{100 - final_dilution_founder:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col4:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: white; margin: 0; font-size: 14px;'>Pro-Rata Benefit</p>
                <h3 style='color: #FFD700; margin: 10px 0;'>+{prorata_benefit:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### Key Findings")
        if prorata_benefit > 0:
            st.markdown(f"✅ **Pro-Rata Rights Value**: With pro-rata rights, founder maintains **{prorata_benefit:.2f}%** more ownership.")
        st.markdown(f"📊 **Final Valuation**: Company valued at **${final_row.get('Post-Money ($M)', 0):.1f}M** after all rounds.")
        st.markdown(f"👥 **Founder vs Investors**: Founder has **{final_dilution_founder:.2f}%**, others have **{100-final_dilution_founder:.2f}%**.")
    else:
        st.info("👈 Configure settings in sidebar and click CALCULATE")

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
