# Mountain Path Design System

## Overview

The Mountain Path Design System provides consistent visual and interaction guidelines for the ARIMA Forecasting Dashboard.

### Brand Philosophy

- **Professional:** Finance industry standard
- **Accessible:** Easy to read and navigate
- **Educational:** Clear information hierarchy
- **Trustworthy:** Reliable and credible appearance

## Color Palette

### Primary Colors

| Color | Hex | RGB | Usage |
|-------|-----|-----|-------|
| Dark Blue | `#003366` | (0, 51, 102) | Primary elements, text |
| Light Blue | `#004d80` | (0, 77, 128) | Secondary elements, hover |
| Gold | `#FFD700` | (255, 215, 0) | Accent, highlights |

### Secondary Colors

- **Light Gray Background:** `#F0F0F0` - Light areas
- **White:** `#FFFFFF` - Main background
- **Red:** `#FF6B6B` - Errors, warnings
- **Green:** `#4CAF50` - Success, positive

### Usage Examples

```python
# src/config.py
DARK_BLUE = "#003366"
LIGHT_BLUE = "#004d80"
GOLD_COLOR = "#FFD700"

# Plotly charts
fig.update_layout(
    plot_bgcolor='#F0F0F0',
    font=dict(color='#003366')
)

# Streamlit elements
st.markdown(
    """<style>
    .stButton>button { color: #003366; }
    </style>""",
    unsafe_allow_html=True
)
```

## Typography

### Font Family

- **Primary Font:** Arial, Sans Serif
- **Fallback:** System sans-serif

### Font Sizes

| Element | Size | Weight |
|---------|------|--------|
| Heading 1 | 24px | Bold |
| Heading 2 | 20px | Bold |
| Body Text | 14px | Regular |
| Small Text | 12px | Regular |
| Code | 13px | Monospace |

### Examples

```markdown
# Heading 1
## Heading 2
Regular body text at 14px
```

## Spacing

### Padding and Margins

- **Extra Small:** 4px
- **Small:** 8px
- **Medium:** 16px
- **Large:** 24px
- **Extra Large:** 32px

### Grid System

- **Base Unit:** 8px
- **Column Width:** 120px
- **Gutter:** 16px

## Components

### Buttons

```html
<!-- Primary Button (Dark Blue) -->
<button style="background-color: #003366; color: white;">Action</button>

<!-- Secondary Button (Light Blue) -->
<button style="background-color: #004d80; color: white;">Secondary</button>

<!-- Accent Button (Gold) -->
<button style="background-color: #FFD700; color: #003366;">Highlight</button>
```

### Cards

```css
.card {
    background-color: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #E0E0E0;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
```

### Forms

- **Input Fields:** Light gray background, dark blue border on focus
- **Labels:** Dark blue text, bold
- **Placeholders:** Light gray text

## Charts & Visualizations

### Chart Styling

```python
import plotly.graph_objects as go

fig = go.Figure()

fig.update_layout(
    # Background
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    paper_bgcolor='white',
    
    # Text
    font=dict(family="Arial", size=12, color='#003366'),
    
    # Grid
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E0E0E0'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E0E0E0'),
    
    # Legend
    legend=dict(bgcolor='rgba(255, 255, 255, 0.8)')
)

# Series colors
fig.add_trace(go.Scatter(
    y=actual_data,
    name='Actual',
    line=dict(color='#003366', width=2.5)
))

fig.add_trace(go.Scatter(
    y=forecast_data,
    name='Forecast',
    line=dict(color='#FF6B6B', width=2.5)
))
```

## Accessibility

### Color Contrast

Minimum contrast ratios:
- Normal text: 4.5:1 with background
- Large text: 3:1 with background

### Dark Blue on White
- **WCAG AA:** ✅ Compliant (ratio: 8.59:1)
- **WCAG AAA:** ✅ Compliant

### Icons

- **Size:** 20px × 20px minimum
- **Color:** Solid dark blue or white

## Responsive Design

### Breakpoints

| Device | Width |
|--------|-------|
| Mobile | 320px - 480px |
| Tablet | 481px - 768px |
| Desktop | 769px + |

### Streamlit Responsive

```python
# Streamlit automatically handles responsive layout
col1, col2 = st.columns(2)
with col1:
    st.metric("Metric 1", value)
with col2:
    st.metric("Metric 2", value)
```

## Interaction States

### Button States

- **Default:** Dark blue, normal
- **Hover:** Light blue, slight elevation
- **Active:** Dark blue, pressed appearance
- **Disabled:** Gray, reduced opacity

### Form States

- **Valid:** Green border
- **Invalid:** Red border
- **Disabled:** Gray background, disabled cursor

## Code Example: Complete Styling

```python
# streamlit config styling
import streamlit as st

# Custom CSS
st.markdown("""
<style>
    /* Primary color for main elements */
    .stButton>button {
        background-color: #003366;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #004d80;
    }
    
    /* Metric styling */
    .metric-card {
        background-color: #FFFFFF;
        border-left: 4px solid #FFD700;
        padding: 16px;
        border-radius: 4px;
    }
    
    /* Success message */
    .success {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 12px;
        border-radius: 4px;
    }
    
    /* Warning message */
    .warning {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 12px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)
```

## Brand Assets

### Logo

File: `assets/mountain-path-logo.png`

Usage:
- Header/navigation
- Favicon
- Documentation

### Icons

Location: `assets/icons/`

Naming convention: `icon-name.svg`

Examples:
- `icon-chart.svg`
- `icon-settings.svg`
- `icon-download.svg`

## Guidelines Summary

✅ **Do:**
- Use primary colors (dark blue) for main elements
- Maintain consistent spacing (8px grid)
- Ensure accessible color contrast
- Keep typography consistent
- Use accent color (gold) sparingly

❌ **Don't:**
- Mix incompatible colors without purpose
- Use oversized fonts
- Create inconsistent spacing
- Use bright or neon colors
- Ignore accessibility standards

## Questions?

For design questions or updates, refer to the design documentation or contact the design team.

---

**Design System Version:** 1.0  
**Last Updated:** January 2026
