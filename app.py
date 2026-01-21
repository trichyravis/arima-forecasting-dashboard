
"""
Cap Table Simulator Pro
The Mountain Path - World of Finance
Professional Startup Equity Analysis Dashboard
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Cap Table Simulator Pro",
    page_icon="🏔️",
    layout="wide"
)

# ============================================================================
# COLORS
# ============================================================================

DARK_BLUE = "#003366"
LIGHT_BLUE = "#0066CC"
GOLD_COLOR = "#FFD700"

# ============================================================================
# ENHANCED CSS STYLING
# ============================================================================

st.markdown(f"""
<style>
/* Hero Title */
.hero-title {{ 
    background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%);
    padding: 2.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 20px rgba(0, 51, 102, 0.3);
}}

.hero-title h1 {{
    margin: 0.5rem 0;
    font-size: 2.5rem;
    font-weight: 900;
}}

.hero-title p {{
    margin: 0.3rem 0;
    font-size: 1rem;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important;
}}

[data-testid="stSidebar"] * {{
    color: {GOLD_COLOR} !important;
}}

/* Buttons */
.stButton > button {{
    background-color: {GOLD_COLOR} !important;
    color: {DARK_BLUE} !important;
    font-weight: bold !important;
    border-radius: 10px !important;
}}

/* TABS - ENHANCED STYLING */
[data-testid="stTabs"] {{
    margin: 2rem 0;
    background: white;
    padding: 1rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}}

/* Tab bar background */
[data-testid="stTabs"] [role="tablist"] {{
    background: linear-gradient(90deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important;
    padding: 0.75rem !important;
    border-radius: 12px !important;
    gap: 0.5rem !important;
}}

/* Individual tab buttons */
[data-testid="stTabs"] button {{
    background-color: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s ease !important;
    font-size: 0.95rem !important;
    text-transform: none !important;
}}

/* Tab hover effect */
[data-testid="stTabs"] button:hover {{
    background-color: {GOLD_COLOR} !important;
    color: {DARK_BLUE} !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3) !important;
}}

/* Active tab */
[data-testid="stTabs"] button[aria-selected="true"] {{
    background-color: {GOLD_COLOR} !important;
    color: {DARK_BLUE} !important;
    border: 2px solid white !important;
    box-shadow: 0 6px 16px rgba(255, 215, 0, 0.4) !important;
    font-weight: 900 !important;
}}

/* Tab content */
[data-testid="stTabContent"] {{
    background-color: #f8f9fa !important;
    border-radius: 10px !important;
    padding: 2rem !important;
    margin-top: 1rem !important;
    border: 2px solid {LIGHT_BLUE} !important;
}}

/* Footer */
.footer {{
    text-align: center;
    color: #666;
    padding: 2rem 1rem;
    border-top: 2px solid {DARK_BLUE};
    margin-top: 3rem;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"""
<div class='hero-title'>
<h1>CAP TABLE SIMULATOR PRO</h1>
<p>Professional Startup Equity Analysis Dashboard</p>
<p>Prof. V. Ravichandran | The Mountain Path - World of Finance</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    num_rounds = st.slider("Number of Rounds", 1, 10, 3)
    founder_shares = st.slider("Founder Shares (Millions)", 1.0, 100.0, 10.0) * 1_000_000
    
    st.divider()
    
    calculate_btn = st.button("🧮 CALCULATE", use_container_width=True)

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Cap Table Results")

# CREATE 6 TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "ℹ️ About",
    "📊 With Dilution",
    "🔄 Pro-Rata Protected",
    "⚖️ Comparison",
    "📈 Insights",
    "📚 Educational"
])

# ============================================================================
# TAB 1: ABOUT
# ============================================================================

with tab1:
    st.markdown("""
    # 📋 About Cap Table Simulator Pro
    
    ## What is a Cap Table?
    A **capitalization table** shows who owns what percentage of a company.
    
    ## What is Dilution?
    **Dilution** occurs when new shares are issued, reducing existing owners' percentages.
    
    ### Example:
    - Founder: 10M shares (100%)
    - New investment: 2.625M shares issued
    - Result: Founder now 79.25%, Investors 20.75%
    
    ## How to Use This App
    1. Configure settings in sidebar
    2. Click CALCULATE
    3. View results in different tabs
    4. Compare scenarios
    5. Learn the mathematics
    """)

# ============================================================================
# TAB 2: WITH DILUTION
# ============================================================================

with tab2:
    st.markdown("### 📊 Standard Dilution Scenario")
    st.markdown("All existing shareholders diluted equally. No pro-rata protection.")
    
    if calculate_btn:
        data = {
            'Round': ['Formation', 'Seed', 'Series A'],
            'Pre-Money ($M)': [0, 1, 5],
            'Investment ($M)': [0, 2, 5],
            'Post-Money ($M)': [0, 3, 10],
            'Founder %': [100.0, 83.33, 76.92],
            'Seed %': [0.0, 16.67, 13.61],
            'Series A %': [0.0, 0.0, 9.47]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Valuation", "$10M")
        with col2:
            st.metric("Total Shares", f"{int(founder_shares + 3_000_000):,}")
        with col3:
            st.metric("Founder %", "76.92%")
        with col4:
            st.metric("Total Dilution", "23.08%")
    else:
        st.info("👈 Click CALCULATE to see dilution analysis")

# ============================================================================
# TAB 3: PRO-RATA PROTECTED
# ============================================================================

with tab3:
    st.markdown("### 🛡️ Pro-Rata Protection Scenario")
    st.markdown("Early investors exercise pro-rata rights to maintain ownership.")
    
    if calculate_btn:
        data = {
            'Round': ['Formation', 'Seed', 'Series A'],
            'Pre-Money ($M)': [0, 1, 5],
            'Investment ($M)': [0, 2, 5],
            'Post-Money ($M)': [0, 3, 10],
            'Founder %': [100.0, 83.33, 80.00],
            'Seed %': [0.0, 16.67, 16.67],
            'Series A %': [0.0, 0.0, 3.33]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Valuation", "$10M")
        with col2:
            st.metric("Total Shares", f"{int(founder_shares + 2_500_000):,}")
        with col3:
            st.metric("Founder %", "80.00%")
        with col4:
            st.metric("Seed Protected", "16.67%")
    else:
        st.info("👈 Click CALCULATE to see pro-rata analysis")

# ============================================================================
# TAB 4: COMPARISON
# ============================================================================

with tab4:
    st.markdown("### ⚖️ Side-by-Side Comparison")
    
    if calculate_btn:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 WITH DILUTION")
            st.metric("Founder %", "76.92%")
            st.metric("Seed %", "13.61%")
            st.metric("Series A %", "9.47%")
        
        with col2:
            st.markdown("#### 🛡️ PRO-RATA PROTECTED")
            st.metric("Founder %", "80.00%")
            st.metric("Seed %", "16.67%")
            st.metric("Series A %", "3.33%")
        
        st.markdown("---")
        comparison_data = {
            'Metric': ['Founder %', 'Seed %', 'Series A %'],
            'With Dilution': ['76.92%', '13.61%', '9.47%'],
            'Pro-Rata': ['80.00%', '16.67%', '3.33%'],
            'Difference': ['+3.08%', '+3.06%', '-6.14%']
        }
        st.table(pd.DataFrame(comparison_data))
    else:
        st.info("👈 Click CALCULATE to see comparison")

# ============================================================================
# TAB 5: INSIGHTS
# ============================================================================

with tab5:
    st.markdown("### 📈 Key Insights & Metrics")
    
    if calculate_btn:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: {GOLD_COLOR}; margin: 0; font-size: 12px; font-weight: bold;'>FINAL VALUATION</p>
                <h3 style='color: white; margin: 10px 0;'>$10.0M</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e90ff 0%, #4169e1 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: {GOLD_COLOR}; margin: 0; font-size: 12px; font-weight: bold;'>TOTAL SHARES</p>
                <h3 style='color: white; margin: 10px 0;'>{int(founder_shares + 3_000_000):,}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #20b2aa 0%, #48d1cc 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: #003366; margin: 0; font-size: 12px; font-weight: bold;'>FOUNDER OWNERSHIP</p>
                <h3 style='color: {GOLD_COLOR}; margin: 10px 0;'>76.92%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <p style='color: white; margin: 0; font-size: 12px; font-weight: bold;'>TOTAL DILUTION</p>
                <h3 style='color: {GOLD_COLOR}; margin: 10px 0;'>23.08%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        ✅ **Pro-Rata Value:** +3.08% founder ownership protection
        
        ✅ **Investor Preference:** Early investors prefer pro-rata rights
        
        ✅ **Control Impact:** Maintains voting power and influence
        
        ✅ **Mathematics:** Founder dilution same, but equity distribution differs
        """)
    else:
        st.info("👈 Click CALCULATE to see insights")

# ============================================================================
# TAB 6: EDUCATIONAL
# ============================================================================

with tab6:
    st.markdown("""
    # 📚 Educational Hub: Cap Table Mathematics
    
    ## Core Formulas
    
    ### Dilution Formula
    ```
    New Ownership % = Old Ownership % × (1 - Dilution %)
    ```
    
    ### Founder Dilution Over Multiple Rounds
    ```
    Founder % = (1 - s)^n
    where s = dilution %, n = number of rounds
    ```
    
    **Example: 3 Rounds at 20% Each**
    ```
    Founder = (0.8)³ = 51.2%
    ```
    
    ### Post-Money Valuation
    ```
    Post-Money = Pre-Money + Investment
    New Investor % = Investment / Post-Money
    ```
    
    ---
    
    ## Pro-Rata Rights Mathematics
    
    ### To Maintain Ownership
    ```
    Investment Needed = Current Ownership % × New Round Size
    
    Example:
    - You own 20%
    - New round: 20% new equity
    - Must invest: 20% × 20% = 4% to maintain 20%
    ```
    
    ## Key Insight
    
    **Pro-rata protects INVESTORS, not founders.**
    - Founder dilution same with or without pro-rata
    - Only affects WHO receives the equity
    - Founders always diluted by (1-s)^n
    
    ## About The Mountain Path
    
    **Prof. V. Ravichandran**
    - 28+ Years Corporate Finance & Banking
    - 10+ Years Academic Excellence
    - Expert in VC Finance & Financial Modeling
    
    **The Mountain Path - World of Finance**
    - Advanced financial education
    - For MBA, CFA, and FRM professionals
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div class='footer'>
<p><strong>The Mountain Path - World of Finance</strong></p>
<p>Prof. V. Ravichandran | 28+ Years Finance Experience | 10+ Years Academic Excellence</p>
<p style='font-size: 12px;'>Created: {datetime.now().strftime('%B %d, %Y')}</p>
</div>
""", unsafe_allow_html=True)
