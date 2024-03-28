import joblib
from joblib import load
import pandas as pd
import modules.prepare_data as prepare_data
import streamlit as st
from streamlit_mic_recorder import mic_recorder

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
    provided_text = "Opera refers to a dramatic art form, originating in Europe."
    with st.container():
        st.write(provided_text)
     
def callback():
    if st.session_state.my_recorder_output:
        try:
            audio_bytes=st.session_state.my_recorder_output['bytes']
            with open('myfile2.wav', mode='wb') as f:
                f.write(audio_bytes)
            st.audio(audio_bytes)
            data = prepare_data.load_audio(provided_text,'myfile2.wav')
            
            #use previous scaler to scale the new prediction to fit into the model
            data = pd.DataFrame([data])
            real_and_transcribed_words_ipa = data['real_and_transcribed_words_ipa'].values[0]
            data = data.drop(columns=['wav_file_path','real_and_transcribed_words_ipa'])

            #####
            def generate_comparison_paragraph(pairs):
                actual_text = " ".join([pair[0] for pair in pairs])
                spoken_text = " ".join([pair[1] for pair in pairs])
                
                # Initialize an empty string for the comparison
                comparison_text = ""
                for actual, spoken in pairs:
                    if actual != spoken:
                        comparison_text += f"<u>{actual}</u> "  # Underline mismatches
                    else:
                        comparison_text += actual + " "  # No underline for matches
                
                return actual_text, spoken_text, comparison_text

            # Generate the texts
            actual_text, spoken_text, comparison_text = generate_comparison_paragraph(real_and_transcribed_words_ipa)

            # Display the paragraphs in Streamlit
            st.markdown("### Actual Text")
            st.markdown(actual_text, unsafe_allow_html=True)

            st.markdown("### Spoken Text")
            st.markdown(spoken_text, unsafe_allow_html=True)

            st.markdown("### Comparison (with underlines for mismatches)")
            st.markdown(comparison_text, unsafe_allow_html=True)
        
            scaler = joblib.load('models/scaler.joblib')
            new_data_normalized = scaler.transform(data)

            # Load the model from the file
            loaded_model = load('models/grid_search_model.joblib')
            # Use the loaded model
            predictions = loaded_model.predict(new_data_normalized) # Assuming you have an X_test set
            data.at[0,'predicted_fluency'] = predictions
            data.columns = ['Words per second','Pause Rate (%)','Pronunciation Accuracy (%)','Fluency']
            html = data.to_html(index=False)
            st.title('Scores:')
            # Use markdown to display the table without index
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            print(e)

mic_recorder(key='my_recorder', callback=callback)
