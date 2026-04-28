# streamlit_app.py
import streamlit as st
# import os
# from dotenv import load_dotenv
# # from places_api import geocode_address, search_and_process
# from datetime import datetime

# load_dotenv()

# API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
# if API_KEY:
#     st.success("API_KEY loaded")
# else:
#     st.error("API_KEY not found; please check your .env file and ensure GOOGLE_PLACES_API_KEY is set")

st.set_page_config(page_title="FocusPlaces", layout="wide")
st.title("FocusPlaces")

st.markdown(
    """
    <div style="margin-top:-8px; margin-bottom:18px;">
      <p style="font-size:18px; color:#6b7280; margin:0; max-width:1200px;">
        Start your search for the best nearby study spots, ranked from real user reviews to help you focus.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Use a full-width container for controls with two side-by-side containers
# We create two container blocks and place them inside a horizontal layout using columns.
# Give them generous relative widths and add spacing via an empty column in-between.
left_container, spacer, right_container = st.columns([1.15, 0.05, 0.95])

with left_container:
    st.header("Search parameters")
    # Wrap controls in a container to keep them grouped
    with st.container():
        queries_text = st.text_area(
            "Search queries (comma separated)",
            value="coffee shop, library, co-working space",
            height=160,
        )
        st.caption("If left blank, default queries (coffee shop, library, co‑working space) will be used.")
        location_input = st.text_input("Location (address) — optional", value="")
        st.caption("If left empty, FocusPlaces will attempt to use current location if available.")
        radius_miles = st.number_input(
            "Radius (miles)",
            min_value=0.1,
            max_value=50.0,
            value=7.5,
            step=0.1,
            format="%.1f",
        )
        radius = int(radius_miles * 1609.344)

with right_container:
    # Put related numeric settings in their own visual group
    st.header("Advanced options")
    with st.container():
        recent_days = st.number_input(
            "Recent time window (days)",
            min_value=30,
            max_value=3650,
            value=900,
        )
        st.caption("Only reviews from the last N days are counted as 'recent' when computing recent-review statistics and the focus score.")
        min_recent = st.number_input(
            "Minimum recent reviews (warning threshold)",
            min_value=1,
            max_value=20,
            value=3,
        )
        st.caption("If a place has fewer than this many recent reviews, results may be less reliable.")
        max_candidates = st.number_input("Max candidates per query", min_value=1, max_value=50, value=5)
        max_reviews_per_place = st.number_input("Max reviews per place to fetch", min_value=1, max_value=20, value=5)

# Action row below the paired containers spanning full width
action_cols = st.columns([1, 4, 1])
with action_cols[1]:
    run = st.button("Run search", use_container_width=True)
    st.markdown("<small style='color:#6b7280;'>Tip: adjust parameters to broaden or narrow your results.</small>", unsafe_allow_html=True)

# Results area uses full width below controls
results_area = st.container()

queries = [q.strip() for q in queries_text.split(",") if q.strip()]
if not queries:
    queries = ["coffee shop", "library", "co-working space"]

if run:
    with st.spinner("Search functionality is disabled."):
        st.info("This demo preserves the app layout and controls, but no Google API calls are made.")
        st.write("The search button was pressed, but actual place lookup and review processing are disabled.")

    with results_area:
        st.info("No results are available because the Google Places API integration has been removed.")
