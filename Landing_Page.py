import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Campaign Analysis App 👋")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 
    
    ### Documentation should go in here
    - Info about the link [text in the link](https://github.com/streamlit/demo-self-driving)
"""
)

url = 'Data_Load'
#st.write("check out this [link](%s)" % url)
st.markdown("Go to the [Data Load Page](%s)" % url)