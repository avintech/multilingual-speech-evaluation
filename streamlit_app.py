import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text

# Page configuration
st.set_page_config(page_title="Speech Evaluation")

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Reading", "Picture Discussion"])

# Page: Image
if selection == "Picture Discussion":
    st.title("Picture Discussion Task")
    st.write("Please describe the below image to the best of your ability.")
    with st.container():
        st.image("images/image1.png")

# Page: Text
elif selection == "Reading":
    st.title("Reading Task")
    st.write("Please read the below text to the best of your ability.")
    with st.container():
        st.write("Opera refers to a dramatic art form, originating in Europe, in which the emotional content is conveyed to the audience as much through music, both vocal and instrumental, as it is through the lyrics. By contrast, in musical theater an actor's dramatic performance is primary, and the music plays a lesser role. The drama in opera is presented using the primary elements of theater such as scenery, costumes, and acting. However, the words of the opera, or libretto, are sung rather than spoken. The singers are accompanied by a musical ensemble ranging from a small instrumental ensemble to a full symphonic orchestra.")
     
def callback():
    if st.session_state.my_recorder_output:
        audio_bytes=st.session_state.my_recorder_output['bytes']
        with open('myfile.wav', mode='bx') as f:
            f.write(audio_bytes)
        st.audio(audio_bytes)
mic_recorder(key='my_recorder', callback=callback)
