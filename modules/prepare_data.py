import eng_to_ipa
import numpy as np
import parselmouth
import scipy.signal
from string import punctuation
import whisper_timestamped as whispert

from modules.WordMatching import get_best_mapped_words
from modules.WordMetrics import edit_distance_python
from modules.zaf import wavread,melfilterbank,mfcc

from dragonmapper import hanzi
import dragonmapper
import torch
import torchaudio
import epitran
import librosa
from sklearn.preprocessing import StandardScaler
            

def load_audio(language,provided_text,audio_path):
    try:
        speech_rate = 0
        pause_rate = 0
        pronunciation_accuracy = 0
        real_and_transcribed_words_ipa = []
        
        audio_data = audio_path['array']
        sample_rate = audio_path['sampling_rate']

        if sample_rate != 16000:
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
            audio_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_tensor)
        else:
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        audio_np = audio_tensor.numpy()

        audio_np = whispert.pad_or_trim(audio_np)

        match language:
            case "zh":
                model = "avintech/whisper-small-chinese"
            case "ms":
                model = "avintech/whisper-small-malay"
            case "ta":
                model = "avintech/whisper-small-tamil"
            case _:
                model = whispert.load_model("base")
        result = whispert.transcribe(model, audio_np, language=language, detect_disfluencies=True)
        recorded_audio_text = result["text"]
        words_list = []
        pause_list = []
        for segment in result["segments"]:
            for words in segment["words"]:
                if words['text'] != "[*]":  
                    #get recognised words
                    words_list.append(words)
                else:
                    #get pauses
                    pause_list.append(words)
        #compute pronunciation accuracy
        real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices = matchWords(language,provided_text, recorded_audio_text)
        pronunciation_accuracy, current_words_pronunciation_accuracy = getPronunciationAccuracy(real_and_transcribed_words_ipa)

        #compute speech rate
        total_duration = result["segments"][-1]["end"]
        speech_rate = calculate_speech_rate(words_list, total_duration)

        #compute pause rate
        pause_rate = get_pause_rate(pause_list,total_duration)

        #compute MFCC
        mfcc = get_mfcc(audio_data, sample_rate,result['segments'])
    except Exception as ex:
        print(ex)
    
    finally:
        data = {
            'real_and_transcribed_words': real_and_transcribed_words,
            'wav_file_path': audio_path,
            'speech_rate': speech_rate,
            'pause_rate': pause_rate,
            'pronunciation_accuracy': pronunciation_accuracy,
            'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa,
            'mfcc': mfcc,
            #'mean_pitch': mean_pitch,
            #'pitch_range': pitch_range,
            #'std_pitch': std_pitch,
        }
        return data

def matchWords(language,provided_text, recorded_transcript):
    #recorded_transcript = recorded_transcript.replace(" ","")
    def convertToPhonem(language: str ,sentence: str) -> str:
        match language:
            case "zh":
                phonem_representation = dragonmapper.hanzi.to_pinyin(sentence)
                phonem_representation = dragonmapper.transcriptions.pinyin_to_ipa(phonem_representation)
            case "ms":
                epi = epitran.Epitran('msa-Latn')  # Replace 'lang_code' with the ISO language code of the language you're interested in
                phonem_representation = epi.transliterate(sentence)
            case "ta":
                epi = epitran.Epitran('tam-Taml')
                phonem_representation = epi.transliterate(sentence)
            case _:
                phonem_representation = eng_to_ipa.convert(sentence)   
        phonem_representation = phonem_representation.replace('*','')
        return phonem_representation
    
    words_estimated = recorded_transcript.split()
    print(words_estimated)
    words_real = provided_text.split()

    mapped_words, mapped_words_indices = get_best_mapped_words(
        words_estimated, words_real)

    real_and_transcribed_words = []
    real_and_transcribed_words_ipa = []
    for word_idx in range(len(words_real)):
        if word_idx >= len(mapped_words)-1:
            mapped_words.append('-')
        real_and_transcribed_words.append(
            (words_real[word_idx],    mapped_words[word_idx]))
        real_and_transcribed_words_ipa.append((convertToPhonem(language, words_real[word_idx]),
                                            convertToPhonem(language, mapped_words[word_idx])))
    return real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices

def getPronunciationAccuracy(real_and_transcribed_words_ipa) -> float:
    def removePunctuation(word: str) -> str:
        return ''.join([char for char in word if char not in punctuation])
    
    total_mismatches = 0.
    number_of_phonemes = 0.
    current_words_pronunciation_accuracy = []
    for pair in real_and_transcribed_words_ipa:
        real_without_punctuation = removePunctuation(pair[0]).lower()
        number_of_word_mismatches = edit_distance_python(
            real_without_punctuation, removePunctuation(pair[1]).lower())
        total_mismatches += number_of_word_mismatches
        number_of_phonemes_in_word = len(real_without_punctuation)
        number_of_phonemes += number_of_phonemes_in_word

        current_words_pronunciation_accuracy.append(float(
            number_of_phonemes_in_word-number_of_word_mismatches)/number_of_phonemes_in_word*100)

    percentage_of_correct_pronunciations = (
        number_of_phonemes-total_mismatches)/number_of_phonemes*100

    return np.round(percentage_of_correct_pronunciations), current_words_pronunciation_accuracy

def calculate_speech_rate(words_list, total_duration):
    total_words = len(words_list)
    
    # Calculate speech rate in words per second
    speech_rate_wps = total_words / total_duration
    
    return speech_rate_wps

def get_pause_rate(pause_list,total_duration):
    total_pause_duration = 0
    for segment in pause_list:
        total_pause_duration += segment['end'] - segment['start']
    pause_rate = (total_pause_duration / total_duration)*100
    return pause_rate

def get_pitch(audio_path):
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = float('nan')
    mean_pitch = np.nanmean(pitch_values)
    pitch_range = np.nanmax(pitch_values) - np.nanmin(pitch_values)
    std_pitch = np.nanstd(pitch_values)

    return mean_pitch,pitch_range,std_pitch

def get_mfcc(y, sr, segments):
    
    # Function to extract MFCC features for a given segment
    def extract_mfcc(y, sr, start, end, n_mfcc=13):
        # Extract the segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        return mfccs

    # Initialize a list to hold the MFCCs for each segment
    all_mfccs = []

    # Extract MFCCs for each speech segment
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        mfccs = extract_mfcc(y, sr, start_time, end_time)
        all_mfccs.append(mfccs)

    def pad_or_truncate_mfccs(mfccs_list, max_length):
        padded_mfccs = []
        for mfcc in mfccs_list:
            if mfcc.shape[1] > max_length:
                # Truncate to max_length
                mfcc = mfcc[:, :max_length]
            else:
                # Pad to max_length
                mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
            padded_mfccs.append(mfcc)
        return np.array(padded_mfccs)
    
    # Define a maximum length for padding/truncating
    max_length = 100

    # Pad or truncate MFCCs to a fixed length
    padded_mfccs = pad_or_truncate_mfccs(all_mfccs, max_length)

    # Normalize padded MFCCs
    scaler = StandardScaler()
    normalized_mfccs = scaler.fit_transform(padded_mfccs.reshape(-1, padded_mfccs.shape[-1])).reshape(padded_mfccs.shape)
    return normalized_mfccs