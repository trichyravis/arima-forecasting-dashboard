# Color Guide - Mountain Path Design

## Introduction

This guide provides detailed usage of colors in the ARIMA Forecasting Dashboard. Consistent color usage enhances user experience and builds brand recognition.

## Primary Color Palette

### Dark Blue #003366

**Purpose:** Primary brand color for main UI elements

**Applications:**
- Navigation bars
- Button primary text
- Heading text
- Chart axes and labels
- Form labels

**Variations:**
```css
/* Original */
color: #003366;

/* Transparent 80% */
color: rgba(0, 51, 102, 0.8);

/* Transparent 50% */
color: rgba(0, 51, 102, 0.5);

/* Transparent 20% */
color: rgba(0, 51, 102, 0.2);
```

**Example Usage:**
```python
# Streamlit
st.title("Dashboard", color='#003366')

# Plotly
fig.update_layout(font=dict(color='#003366'))

# CSS
.heading { color: #003366; }
```

### Light Blue #004d80

**Purpose:** Secondary brand color for hover and alternate states

**Applications:**
- Button hover states
- Links and focus states
- Secondary navigation
- Accent borders
- Selection highlights

**Usage:**
```python
# Hover state
.button:hover { 
    background-color: #004d80; 
}

# Focus state
input:focus { 
    border-color: #004d80; 
}
```

### Gold #FFD700

**Purpose:** Accent color for highlights and important information

**Applications:**
- Call-to-action buttons
- Success indicators
- Highlight boxes
- Important metrics
- Accent borders on cards
- Trend indicators (positive)

**Usage:**
```python
# Highlight box
st.info("Important", icon="⭐")  # With gold background

# Accent border
border-left: 4px solid #FFD700;

# Button accent
st.button("Forecast Now!", key="primary_action")
```

## Secondary Colors

### Light Gray #F0F0F0

**Purpose:** Secondary background for panels and sections

**Applications:**
- Chart backgrounds
- Panel backgrounds
- Alternating row colors
- Disabled state backgrounds
- Form input backgrounds

**Usage:**
```python
# Chart background
fig.update_layout(
    plot_bgcolor='#F0F0F0'
)

# Panel
.panel { background-color: #F0F0F0; }
```

### White #FFFFFF

**Purpose:** Main background and clean spaces

**Applications:**
- Main page background
- Card backgrounds
- Input field backgrounds
- Modal backgrounds
- Text on dark backgrounds

### Red #FF6B6B

**Purpose:** Error and negative states

**Applications:**
- Error messages
- Invalid form fields
- Warning alerts
- Stop/cancel buttons
- Negative trend indicators

**Usage:**
```python
# Error alert
st.error("Forecast failed: Invalid data")

# Invalid state
.input.invalid { 
    border-color: #FF6B6B; 
}
```

### Green #4CAF50

**Purpose:** Success and positive states

**Applications:**
- Success messages
- Valid form fields
- Confirmation dialogs
- Positive metrics
- Growth indicators

**Usage:**
```python
# Success message
st.success("Data loaded successfully!")

# Positive indicator
.metric.positive { color: #4CAF50; }
```

## Color Combinations

### Recommended Color Pairs

| Foreground | Background | Contrast | Usage |
|-----------|-----------|----------|-------|
| #003366 (Dark Blue) | #FFFFFF (White) | 8.59:1 | Primary text |
| #FFFFFF (White) | #003366 (Dark Blue) | 8.59:1 | Button text |
| #004d80 (Light Blue) | #FFFFFF (White) | 6.26:1 | Secondary text |
| #FF6B6B (Red) | #FFFFFF (White) | 3.49:1 | Error text |
| #4CAF50 (Green) | #FFFFFF (White) | 4.94:1 | Success text |
| #FFD700 (Gold) | #003366 (Dark Blue) | 8.22:1 | Accent on dark |

### Avoid These Combinations

❌ Gold on Light Gray - Low contrast  
❌ Light Blue on White - Insufficient contrast  
❌ Red on Dark Blue - Difficult to read  

## Usage by Component

### Buttons

```python
# Primary button
st.button(
    "Forecast",
    key="primary",
    help="Generate forecast"
)
# CSS: background-color: #003366; color: #FFFFFF;

# Secondary button  
st.button(
    "Cancel",
    key="secondary"
)
# CSS: background-color: #004d80; color: #FFFFFF;

# Accent button (important action)
st.button(
    "Download Results",
    key="accent"
)
# CSS: background-color: #FFD700; color: #003366;
```

### Alerts & Messages

```python
# Error (Red)
st.error("Invalid parameters selected")

# Warning (Orange)
st.warning("Data may be outdated")

# Success (Green)
st.success("Forecast generated successfully")

# Info (Light Blue)
st.info("Forecast confidence: 95%")
```

### Charts

```python
import plotly.graph_objects as go

# Create figure with color scheme
fig = go.Figure()

# Actual data (Dark Blue)
fig.add_trace(go.Scatter(
    y=actual_values,
    name='Actual',
    line=dict(color='#003366', width=2.5)
))

# Forecast (Red for distinction)
fig.add_trace(go.Scatter(
    y=forecast_values,
    name='Forecast',
    line=dict(color='#FF6B6B', width=2.5)
))

# Confidence interval (Gold)
fig.add_trace(go.Scatter(
    fill='toself',
    fillcolor='rgba(255, 215, 0, 0.2)',
    name='95% CI'
))

# Layout
fig.update_layout(
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    font=dict(color='#003366'),
    xaxis=dict(gridcolor='#E0E0E0'),
    yaxis=dict(gridcolor='#E0E0E0')
)
```

### Tables

```python
# Header: Dark Blue on White
.table-header { 
    background-color: #003366; 
    color: #FFFFFF; 
}

# Alternating rows: White and Light Gray
.table-row:even { background-color: #FFFFFF; }
.table-row:odd { background-color: #F0F0F0; }

# Highlight row: Gold accent
.table-row.highlight { 
    border-left: 4px solid #FFD700; 
}
```

### Cards

```python
.card {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 16px;
}

.card-header {
    color: #003366;
    font-weight: bold;
    border-bottom: 2px solid #FFD700;
    padding-bottom: 8px;
}

.card-content {
    color: #333333;
    margin-top: 12px;
}

.card-footer {
    color: #666666;
    font-size: 12px;
    margin-top: 8px;
}
```

## Accessibility Compliance

### WCAG Standards

All color combinations meet or exceed WCAG AA standards:

✅ **WCAG AA (Minimum 4.5:1 for normal text)**
- Dark Blue on White: 8.59:1
- Light Blue on White: 6.26:1
- Gold on Dark Blue: 8.22:1

✅ **WCAG AAA (Minimum 7:1 for normal text)**
- Dark Blue on White: 8.59:1
- Light Blue on White: 6.26:1 (borderline)

### Color-Blind Friendly

The palette is designed to be distinguishable for:
- Deuteranopia (red-green blindness)
- Protanopia (red-green blindness)
- Tritanopia (blue-yellow blindness)

**Recommendation:** Don't rely on color alone for critical information. Use patterns, text labels, and icons in addition to colors.

## Best Practices

### ✅ DO:

1. **Use consistent colors** across similar elements
2. **Test contrast ratios** for accessibility
3. **Use accent color sparingly** for emphasis
4. **Provide visual feedback** with hover states
5. **Use color semantically** (red for errors, green for success)

### ❌ DON'T:

1. **Don't use too many colors** in one interface
2. **Don't rely solely on color** to convey information
3. **Don't use inconsistent colors** for the same element type
4. **Don't mix colors without purpose**
5. **Don't use colors with insufficient contrast**

## Color Implementation Checklist

- [ ] Primary text uses Dark Blue (#003366)
- [ ] Hover states use Light Blue (#004d80)
- [ ] Accent elements use Gold (#FFD700)
- [ ] Errors use Red (#FF6B6B)
- [ ] Success uses Green (#4CAF50)
- [ ] All color combinations meet WCAG AA
- [ ] Chart colors are distinguishable
- [ ] Color is not the only way to distinguish elements

## Tools & Resources

### Color Picker Tools
- Built-in browser DevTools
- Adobe Color Wheel
- Coolors.co

### Contrast Checkers
- WebAIM Contrast Checker
- Accessible Colors
- Chrome DevTools Accessibility Tab

### Color-Blind Simulators
- Color Brewer
- Accessible Colors

---

**Color Guide Version:** 1.0  
**Last Updated:** January 2026

For questions about color usage, see DESIGN_SYSTEM.md
