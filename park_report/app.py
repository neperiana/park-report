"""
# Park Report
Dashboard that visualises parkrun stats for one specific parkrun event.
"""

from os.path import exists
import time

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

def validate_parkrun_event(park_event_id):
    return exists(f'data/{park_event_id}.csv')

def get_data(park_event_id):
    return pd.read_csv(f'data/{park_event_id}.csv')

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        logo = Image.open('park-report-logo.png')
        st.image(logo)

    with col2:
        park_event_id = st.text_input(
            "Enter parkrun event id 👇",
            value="philipspark",
        )
        is_parkrun_valid = validate_parkrun_event(park_event_id)

        if is_parkrun_valid:
            st.success(f'Parkrun {park_event_id} found', icon="✅")
            
        else:
            st.error(f'Parkrun {park_event_id} not found', icon="🚨")
            st.stop()

with st.spinner('Loading data...'):
    time.sleep(2)
    df = get_data(park_event_id)


participants = df.groupby('event_no')[['position']].max()
participants = participants.rename(columns={'position': 'participants'})

st.line_chart(data=participants)