import streamlit as st
from helpers import *
import asyncio
from st_pages import get_nav_from_toml


async def run_async_user_rationale(placeholder7, user_commit):
  # TODO: preprocess commit and apply classifiers to it , output the classfication results
   user_commit_rationale_density = await commit_density(user_commit)
   with placeholder7.container():
       st.write( f"Your commit message's rationale density: {user_commit_rationale_density:.2%}")

############### Page
async def user_commit_page():
    st.set_page_config(page_title="Linux Rationale Analyses Tool")
    st.header("Commit Message Rationale Analyses")
    
    user_commit = st.text_input("Enter your commit message","")
    submit = st.button('Start Commit Analysis')  
    if submit: 
        placeholder7 =  st.empty()
        with placeholder7 : st.markdown(
                """<div style='text-align:center; font-size:24px; color:gray;'>Loading... ⏳</div>""",
                    unsafe_allow_html=True,
            )
        asyncio.create_task(run_async_user_rationale(placeholder7, user_commit)    )



##############
if __name__ == '__main__':
    # Streamlit is synchronous, but we can trigger asynchronous tasks with create_task
    asyncio.run(user_commit_page())