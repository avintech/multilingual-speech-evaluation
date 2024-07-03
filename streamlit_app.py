import joblib
from joblib import load
import pandas as pd
import modules.prepare_data as prepare_data
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import keras
import numpy as np
import librosa
from collections import Counter
import numpy as np
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
import re


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
        st.image("data/images/image1.png")
    with open("data/images/image1_desc.txt", 'r', encoding='utf-8') as file:
        reference_text = file.read()
    
    lang = st.selectbox(
    "Choose a language",
    ("Chinese", "Malay", "Tamil"))

    if lang == "Chinese":
        model = '0'
    elif lang == "Malay":
        model = '1'
    elif lang == "Tamil":
        model = '0'
    
    st.write("You selected:", lang)
    st.session_state.lang = lang
    st.session_state.reference_text = reference_text
    st.session_state.model = model

# Page: Text
elif selection == "Reading":
    st.title("Reading Task")
    
    lang = st.selectbox(
    "Choose a language",
    ("Chinese", "Malay", "Tamil"))
    st.write("You selected:", lang)
    st.session_state.lang = lang

    st.write("Please read the below text to the best of your ability.")

    if lang == "Chinese":
        provided_text = "小明和朋友们在公园里玩。他们笑啊，跑啊，玩得很开心。太阳下山了，他们回家了。今天真快乐！"
        model = '0'
    elif lang == "Malay":
        provided_text = "Ali dan kawan-kawannya bermain di taman. Mereka ketawa, berlari dan sangat gembira. Matahari terbenam, mereka pulang. Hari ini sangat indah!"
        model = '1'
    elif lang == "Tamil":
        provided_text = "அலி மற்றும் நண்பர்கள் பூங்காவில் விளையாடினர். அவர்கள் சிரித்து, ஓடி மகிழ்ச்சியாக விளையாடினர். சூரியன் மறைய, அவர்கள் வீட்டுக்கு 戻りました. இன்று மிக அழக照 நாள்!"
        model = '0'
    
    with st.container():
        st.write(provided_text)
     
def reading_callback():
    if st.session_state.my_recorder_output:
        try:
            lang = st.session_state.lang

            audio_bytes=st.session_state.my_recorder_output['bytes']
            with open('myfile2.wav', mode='wb') as f:
                f.write(audio_bytes)
            st.audio(audio_bytes)

            audio_array, sampling_rate = librosa.load('myfile2.wav', sr=None)
            audio_data = {'array': audio_array, 'sampling_rate': sampling_rate}

            if lang == "Chinese":
                model_lang = "zh"
            elif lang == "Malay":
                model_lang = "ms"
            elif lang == "Tamil":
                model_lang = "ta"

            data = prepare_data.load_audio(0,model_lang,provided_text,audio_data)
            #use previous scaler to scale the new prediction to fit into the model
            data = pd.DataFrame([data])
            data['mfcc'] = data['mfcc'].apply(lambda x: x.flatten())
            mfcc_length = data['mfcc'].apply(len).max()
            data['mfcc'] = data['mfcc'].apply(lambda x: np.pad(x, (0, mfcc_length - len(x)), mode='constant'))

            # Convert mfcc column into multiple columns
            mfcc_features = np.stack(data['mfcc'].values)
            df_mfcc = pd.DataFrame(mfcc_features, index=data.index)
            X = pd.concat([data[['speech_rate', 'pause_rate', 'pronunciation_accuracy']], df_mfcc], axis=1)
            X.columns = X.columns.astype(str)

            #Load scalar
            scaler = StandardScaler()
            X_train = pd.read_pickle("data/pickles/"+lang.lower()+"_X_train.pkl")
            scaler.fit(X_train)

            #Normalise new data
            new_data_scaled = scaler.transform(X)

            # Load the model from the file
            # 0 for XGBoost, 1 for Random Forest
            if model == '0':
                loaded_model = keras.models.load_model('models/model_'+lang.lower()+'.keras')
            elif model == '1':
                loaded_model = load('models/random_forest_model_'+lang.lower()+'.joblib')
            else:
                exit

            y_pred = loaded_model.predict(new_data_scaled)
            y_pred_class = np.argmax(y_pred, axis=1)

            print("Fluency Score: " + str(y_pred_class))
#####################################################################################################################################
            real_and_transcribed_words_ipa = data['real_and_transcribed_words_ipa'].values[0]
            real_and_transcribed_words = data['real_and_transcribed_words'].values[0]
            data = data.drop(columns=['wav_file_path','real_and_transcribed_words_ipa'])
            print(real_and_transcribed_words_ipa)
            print(real_and_transcribed_words)

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
            st.markdown("### Actual Phoneme")
            st.markdown(actual_text, unsafe_allow_html=True)

            st.markdown("### Spoken Phoneme")
            st.markdown(spoken_text, unsafe_allow_html=True)

            st.markdown("### Comparison of Phonemes (with underlines for mismatches)")
            st.markdown(comparison_text, unsafe_allow_html=True)

            # Generate the texts
            actual_text, spoken_text, comparison_text = generate_comparison_paragraph(real_and_transcribed_words)

            # Display the paragraphs in Streamlit
            st.markdown("### Actual Text")
            st.markdown(actual_text, unsafe_allow_html=True)

            st.markdown("### Spoken Text")
            st.markdown(spoken_text, unsafe_allow_html=True)

            st.markdown("### Comparison of Text (with underlines for mismatches)")
            st.markdown(comparison_text, unsafe_allow_html=True)
                        
            data.at[0, 'predicted_fluency'] = y_pred_class
            data = data.drop(columns=['real_and_transcribed_words', 'mfcc', 'transcript'])

            # Rename columns
            data.columns = ['Words per second', 'Pause Rate (%)', 'Pronunciation Accuracy (%)', 'Fluency']

            # Convert DataFrame to HTML
            html = data.to_html(index=False)

            # Streamlit display
            st.title('Scores:')
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            print(e)

     
def picture_callback():
    if st.session_state.my_recorder_output:
        try:
            reference_text = st.session_state.reference_text
            lang = st.session_state.lang
            model = st.session_state.model

            #Fluency
            audio_bytes=st.session_state.my_recorder_output['bytes']
            with open('myfile2.wav', mode='wb') as f:
                f.write(audio_bytes)
            st.audio(audio_bytes)

            audio_array, sampling_rate = librosa.load('myfile2.wav', sr=None)
            audio_data = {'array': audio_array, 'sampling_rate': sampling_rate}

            if lang == "Chinese":
                model_lang = "zh"
            elif lang == "Malay":
                model_lang = "ms"
            elif lang == "Tamil":
                model_lang = "ta"
            provided_text = ""
            data = prepare_data.load_audio(1,model_lang,provided_text,audio_data)

            #use previous scaler to scale the new prediction to fit into the model
            data = pd.DataFrame([data])

            transcript = data['transcript'][0]
            print(transcript)
            st.markdown("### What you said:")
            st.markdown(transcript, unsafe_allow_html=True)

            data['mfcc'] = data['mfcc'].apply(lambda x: x.flatten())
            mfcc_length = data['mfcc'].apply(len).max()
            data['mfcc'] = data['mfcc'].apply(lambda x: np.pad(x, (0, mfcc_length - len(x)), mode='constant'))

            # Convert mfcc column into multiple columns
            mfcc_features = np.stack(data['mfcc'].values)
            df_mfcc = pd.DataFrame(mfcc_features, index=data.index)
            X = pd.concat([data[['speech_rate', 'pause_rate', 'pronunciation_accuracy']], df_mfcc], axis=1)
            X.columns = X.columns.astype(str)

            #Load scalar
            scaler = StandardScaler()
            X_train = pd.read_pickle("data/pickles/"+lang.lower()+"_X_train.pkl")
            scaler.fit(X_train)

            #Normalise new data
            new_data_scaled = scaler.transform(X)

            # Load the model from the file
            # 0 for XGBoost, 1 for Random Forest
            if model == '0':
                loaded_model = keras.models.load_model('models/model_'+lang.lower()+'.keras')
            elif model == '1':
                loaded_model = load('models/random_forest_model_'+lang.lower()+'.joblib')
            else:
                exit

            y_pred = loaded_model.predict(new_data_scaled)
            y_pred_class = np.argmax(y_pred, axis=1)

            print("Fluency Score: " + str(y_pred_class))
            st.markdown("### Fluency Score:")
            #show table on pause and speech rate
            fluency_dic = {
                "Speech Rate": f"{data['speech_rate'][0]:.2f}",
                "Pause Rate": f"{data['pause_rate'][0]:.2f}"
            }
            fluency_df = pd.DataFrame(list(fluency_dic.items()),columns=["Metric","Score"])
            fluency_html = fluency_df.to_html(index=False)
            st.markdown(fluency_html, unsafe_allow_html=True)

            
            #Vocab
            words = jieba.lcut(transcript)
            total_words = len(words)
            unique_words = set(words)
            num_unique_words = len(unique_words)
            ttr = num_unique_words / total_words
            rttr = num_unique_words / np.sqrt(total_words)
            hapax_legomena = [word for word, count in Counter(words).items() if count == 1]
            hapax_legomena_ratio = len(hapax_legomena) / total_words
            frequencies = Counter(words).values()
            word_probs = [freq / total_words for freq in frequencies]
            shannon_entropy = -sum(p * np.log2(p) for p in word_probs)
            normalized_ttr = ttr
            normalized_rttr = rttr / np.sqrt(total_words)
            normalized_hapax = hapax_legomena_ratio
            max_entropy = np.log2(num_unique_words) if num_unique_words > 0 else 1
            normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
            vocabulary_richness_score = (
                normalized_ttr + 
                normalized_rttr + 
                normalized_hapax + 
                normalized_entropy
            ) / 4
            vocabulary_richness_percentage = vocabulary_richness_score * 100
            print("总词数:", total_words)
            print("独特词汇数:", num_unique_words)
            print("Type-Token Ratio (TTR):", ttr)
            print("Root Type-Token Ratio (RTTR):", rttr)
            print("Hapax Legomena Ratio:", hapax_legomena_ratio)
            print("Shannon Entropy:", shannon_entropy)
            print(f"Vocabulary Richness Percentage: {vocabulary_richness_percentage:.2f}")

            vocab_richness_data = {
                "Total Words": f"{total_words:.2f}",
                "Unique Words": f"{num_unique_words:.2f}",
                "Type-Token Ratio (TTR)": f"{ttr:.2f}",
                "Root Type-Token Ratio (RTTR)": f"{rttr:.2f}",
                "Hapax Legomena Ratio": f"{hapax_legomena_ratio:.2f}",
                "Shannon Entropy": f"{shannon_entropy:.2f}",
                "Vocabulary Richness Percentage": f"{vocabulary_richness_percentage:.2f}"
            }

            # Convert the dictionary into a DataFrame
            vocab_richness_df = pd.DataFrame(list(vocab_richness_data.items()),columns=["Metric","Score"])
            vocab_richness_html = vocab_richness_df.to_html(index=False)
            st.markdown("### Vocabulary Richness:")
            st.markdown(vocab_richness_html, unsafe_allow_html=True)

            #Similarity
            def segment(text):
                return ' '.join(jieba.cut(text))
            segmented_sentence1 = segment(transcript)
            segmented_sentence2 = segment(reference_text)
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            embedding1 = model.encode(segmented_sentence1)
            embedding2 = model.encode(segmented_sentence2)
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            similarity_score = similarity * 100
            print(f"Semantic Similarity Score: {similarity_score:.2f}")

            #5W1H
            model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")
            prompt = (
                transcript +
                "based on the text, identify what are the 5W1H in Chinese"
            )
            output = model.generate(prompt, temp=0)
            def parse_input(input_str):
                lines = input_str.strip().split('\n')
                details = {}
                for line in lines:
                    if ': ' in line:
                        key, value = line.split(': ', 1)  # Split only on the first occurrence of ': '
                        details[key] = value
                    else:
                        print(f"Skipping malformed line: {line}")
                return details

            def evaluate_details(details):
                if not isinstance(details, dict):
                    raise TypeError("Details should be a dictionary.")
                evaluation = {
                    "Who": 1 if details.get("Who") not in ["不明", "不知道"] else 0,
                    "What": 1 if details.get("What") not in ["不明", "不知道"] else 0,
                    "Where": 1 if details.get("Where") not in ["不明", "不知道"] else 0,
                    "When": 1 if details.get("When") not in ["不明", "不知道"] else 0,
                    "Why": 1 if details.get("Why") not in ["不明", "不知道"] else 0,
                    "How": 1 if details.get("How") not in ["不明", "不知道"] else 0
                }
                total_score = sum(evaluation.values())
                return evaluation, total_score
            input_str = output
            details = parse_input(input_str)
            evaluation, total_score = evaluate_details(details)
            whscore = (total_score/6)*100

            print(f"Evaluation: {evaluation}")
            eval_df = pd.DataFrame(list(evaluation.items()), columns=['Question', 'Score'])
            eval_df['Score'] = eval_df['Score'] == 1
            eval_html = eval_df.to_html(index=False)
            st.markdown("### 5W1H Score:")
            st.markdown(eval_html, unsafe_allow_html=True)
            print(f"5W1H Score: {whscore}")

            #Grammar
            grammar_prompt = (
                transcript + "Based on the sentence, evaluate the grammar by giving a score out of 100"
            )

            # Generate response with specific parameters for consistency
            #temp=0 reduce randomness
            grammar_output = model.generate(grammar_prompt, temp=0)
            print(grammar_output)
            score_match = re.search(r"\d+", grammar_output)
            if score_match:
                grammar_score = int(score_match.group())
                print(f"Score: {grammar_score}")

            data = {
                "Metric": [
                    "Vocabulary Richness",
                    "5W1H",
                    "Content Relevance",
                    "Grammar",
                    "Fluency"
                ],
                "Score": [
                    f"{vocabulary_richness_percentage:.2f}",
                    f"{whscore:.2f}",
                    f"{similarity_score:.2f}",
                    f"{grammar_score:.2f}",
                    f"{y_pred_class[0]:.2f}"
                ]
            }

            # Convert the dictionary into a DataFrame
            overall_score_df = pd.DataFrame(data)
            st.markdown("### Overall Score:")
            st.table(overall_score_df)

            


        except Exception as e:
            print(e)


if selection == "Picture Discussion":
    mic_recorder(key='my_recorder', callback=picture_callback)
elif selection == "Reading":
    mic_recorder(key='my_recorder', callback=reading_callback)
