import streamlit as st
import datetime
import json
import os

# Set page config for a kid-friendly theme
st.set_page_config(
    page_title="Kids' Prayer Times",
    page_icon="🌙",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; color: #FF6B6B; }
    .highlight { background-color: #FFF3CD; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Sample prayer times
prayer_times = {
    "Fajr": "4:00 AM",
    "Dhuhr": "11:55 AM",
    "Asr": "3:30 PM",
    "Maghrib": "6:20 PM",
    "Isha": "7:30 PM"
}

# Emoji/icons
prayer_emojis = {
    "Fajr": "🌙",
    "Dhuhr": "🌞",
    "Asr": "🌤️",
    "Maghrib": "🌆",
    "Isha": "🌃"
}

# Initialize session state
if 'checked' not in st.session_state:
    st.session_state.checked = {prayer: False for prayer in prayer_times}

# Load historical data from file (persists across app restarts)
HISTORY_FILE = "prayer_history.json"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump({"completed_dates": []}, f)

with open(HISTORY_FILE, "r") as f:
    history_data = json.load(f)
    completed_dates = history_data["completed_dates"]

# Track today's date
today = datetime.date.today()
today_str = str(today)

# Title and Header
st.title("🌙 Kids' Prayer Times Tracker")
st.markdown("---")

# Display prayer times with checkboxes
for prayer, time in prayer_times.items():
    col1, col2, col3 = st.columns([1, 3, 3])
    with col1:
        st.markdown(f"<span class='big-font'>{prayer_emojis[prayer]}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**{prayer}**\n\n`{time}`")
    with col3:
        st.session_state.checked[prayer] = st.checkbox(
            f"Done with {prayer}!",
            value=st.session_state.checked[prayer],
            key=prayer
        )

# Progress Tracker
completed = sum(st.session_state.checked.values())
total = len(prayer_times)
progress = completed / total

# Update completed dates (only once per day)
if completed == total and today_str not in completed_dates:
    completed_dates.append(today_str)
    with open(HISTORY_FILE, "w") as f:
        json.dump({"completed_dates": completed_dates}, f)

# Display progress
st.markdown("---")
st.subheader("Today's Progress 🎉")
st.markdown(f"**{completed}/{total} Prayers Completed**")
st.progress(progress)

# Clapping effect if all prayers are done
if completed == total:
    st.balloons()
    st.success("🌟 Amazing! You finished all prayers today!")
    st.write("👏" * 10 + " YAY! " + "👏" * 10)

# Display total days completed (persists across resets)
st.markdown("---")
st.subheader("Total Days Completed")
st.markdown(f"""
    <div class="highlight">
        🏆 **{len(completed_dates)} day{'s' if len(completed_dates) != 1 else ''}** where you prayed all 5 times!
    </div>
""", unsafe_allow_html=True)

# Timeline and Reset button
st.markdown("---")
st.subheader("Day Timeline")
timeline = "🕒 " + " — ".join([f"{prayer_emojis[prayer]} {time}" for prayer, time in prayer_times.items()])
st.markdown(f"`{timeline}`")

if st.button("Reset Checkboxes"):
    st.session_state.checked = {prayer: False for prayer in prayer_times}
    st.rerun()  # Refresh to reflect reset checkboxes