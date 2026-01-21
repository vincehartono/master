import streamlit as st
import json
import os
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="🎰 Roulette Predictor", layout="wide")

# Color mapping
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
HISTORY_FILE = "roulette_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                return data.get('draws', [])[-6:]
        except:
            return []
    return []

def save_history(draws):
    with open(HISTORY_FILE, 'w') as f:
        json.dump({'draws': draws, 'timestamp': datetime.now().isoformat()}, f)

def get_priority_score(category, value, draws):
    """Calculate priority score - lower means higher priority"""
    if not value or value == "Either":
        return 999
    
    if category == "color":
        red_count = sum(1 for d in draws if d in RED_NUMBERS)
        black_count = sum(1 for d in draws if d in BLACK_NUMBERS)
        return red_count if value == "RED" else black_count
    elif category == "parity":
        odd_count = sum(1 for d in draws if d % 2 == 1)
        even_count = sum(1 for d in draws if d % 2 == 0)
        return odd_count if value == "ODD" else even_count
    elif category == "size":
        small_count = sum(1 for d in draws if 1 <= d <= 18)
        big_count = sum(1 for d in draws if 19 <= d <= 36)
        return small_count if value == "SMALL" else big_count
    return 0

def analyze_color(draws):
    if not draws:
        return "No draws yet", ""
    
    red_count = sum(1 for d in draws if d in RED_NUMBERS)
    black_count = sum(1 for d in draws if d in BLACK_NUMBERS)
    
    drawn_red = any(d in RED_NUMBERS for d in draws)
    drawn_black = any(d in BLACK_NUMBERS for d in draws)
    
    recent_draws = draws[-4:] if len(draws) >= 4 else draws
    recent_red = sum(1 for d in recent_draws if d in RED_NUMBERS)
    recent_black = sum(1 for d in recent_draws if d in BLACK_NUMBERS)
    
    if not drawn_red:
        rec = "RED"
    elif not drawn_black:
        rec = "BLACK"
    elif red_count < black_count:
        rec = "RED"
    elif black_count < red_count:
        rec = "BLACK"
    else:
        if recent_red > recent_black:
            rec = "RED"
        elif recent_black > recent_red:
            rec = "BLACK"
        else:
            rec = "Either"
    
    return (red_count, black_count), rec

def analyze_parity(draws):
    if not draws:
        return "No draws yet", ""
    
    odd_count = sum(1 for d in draws if d % 2 == 1)
    even_count = sum(1 for d in draws if d % 2 == 0)
    
    drawn_odd = any(d % 2 == 1 for d in draws)
    drawn_even = any(d % 2 == 0 for d in draws)
    
    recent_draws = draws[-4:] if len(draws) >= 4 else draws
    recent_odd = sum(1 for d in recent_draws if d % 2 == 1)
    recent_even = sum(1 for d in recent_draws if d % 2 == 0)
    
    if not drawn_odd:
        rec = "ODD"
    elif not drawn_even:
        rec = "EVEN"
    elif odd_count < even_count:
        rec = "ODD"
    elif even_count < odd_count:
        rec = "EVEN"
    else:
        if recent_odd > recent_even:
            rec = "ODD"
        elif recent_even > recent_odd:
            rec = "EVEN"
        else:
            rec = "Either"
    
    return (odd_count, even_count), rec

def analyze_size(draws):
    if not draws:
        return "No draws yet", ""
    
    small_count = sum(1 for d in draws if 1 <= d <= 18)
    big_count = sum(1 for d in draws if 19 <= d <= 36)
    
    drawn_small = any(1 <= d <= 18 for d in draws)
    drawn_big = any(19 <= d <= 36 for d in draws)
    
    recent_draws = draws[-4:] if len(draws) >= 4 else draws
    recent_small = sum(1 for d in recent_draws if 1 <= d <= 18)
    recent_big = sum(1 for d in recent_draws if 19 <= d <= 36)
    
    if not drawn_small:
        rec = "SMALL"
    elif not drawn_big:
        rec = "BIG"
    elif small_count < big_count:
        rec = "SMALL"
    elif big_count < small_count:
        rec = "BIG"
    else:
        if recent_small > recent_big:
            rec = "SMALL"
        elif recent_big > recent_small:
            rec = "BIG"
        else:
            rec = "Either"
    
    return (small_count, big_count), rec

def create_pie_chart(labels, values, colors, title):
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='inside',
        hovertemplate='%{label}: %{value}/6<extra></extra>'
    )])
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12)
    )
    return fig

# Initialize session state
if 'draws' not in st.session_state:
    st.session_state.draws = load_history()

# Title
st.title("🎰 ROULETTE PREDICTOR 🎰")

# Input Section
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    num_input = st.number_input("Enter roulette number (0-36):", min_value=0, max_value=36, step=1, key="input")
with col2:
    if st.button("Add Draw"):
        if 0 <= num_input <= 36:
            st.session_state.draws.append(int(num_input))
            if len(st.session_state.draws) > 6:
                st.session_state.draws.pop(0)
            save_history(st.session_state.draws)
            st.rerun()
with col3:
    if st.button("Clear History"):
        st.session_state.draws = []
        save_history([])
        st.rerun()

# Last 6 Draws
if st.session_state.draws:
    st.markdown("### Last 6 Draws")
    draws_text = " → ".join(str(d) for d in st.session_state.draws)
    st.markdown(f"### **{draws_text}**", unsafe_allow_html=True)
else:
    st.info("No draws yet. Add at least 1 draw to see analysis.")
    st.stop()

# Analysis Section
st.markdown("## Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    color_counts, color_rec = analyze_color(st.session_state.draws)
    fig_color = create_pie_chart(
        ["RED", "BLACK"],
        list(color_counts),
        ["#e74c3c", "#000000"],
        "🔴 RED vs ⚫ BLACK"
    )
    st.plotly_chart(fig_color, use_container_width=True)

with col2:
    parity_counts, parity_rec = analyze_parity(st.session_state.draws)
    fig_parity = create_pie_chart(
        ["ODD", "EVEN"],
        list(parity_counts),
        ["#3498db", "#2ecc71"],
        "🔵 ODD vs EVEN"
    )
    st.plotly_chart(fig_parity, use_container_width=True)

with col3:
    size_counts, size_rec = analyze_size(st.session_state.draws)
    fig_size = create_pie_chart(
        ["SMALL (1-18)", "BIG (19-36)"],
        list(size_counts),
        ["#f39c12", "#9b59b6"],
        "📊 SMALL vs BIG"
    )
    st.plotly_chart(fig_size, use_container_width=True)

# Recommendations
st.markdown("## 🎯 Next Pick Recommendations")

recommendations = []
if color_rec:
    recommendations.append(("Color", color_rec, get_priority_score("color", color_rec, st.session_state.draws)))
if parity_rec:
    recommendations.append(("Parity", parity_rec, get_priority_score("parity", parity_rec, st.session_state.draws)))
if size_rec:
    recommendations.append(("Size", size_rec, get_priority_score("size", size_rec, st.session_state.draws)))

recommendations.sort(key=lambda x: x[2])

cols = st.columns(len(recommendations))
for i, (col, (category, rec, score)) in enumerate(zip(cols, recommendations)):
    with col:
        st.metric(label=category, value=rec)

# Display as ranked list
st.markdown("### By Priority (Most Urgent):")
for i, (category, rec, score) in enumerate(recommendations, 1):
    st.markdown(f"**{i}. {rec}**")
