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
            


def load_audio(language, provided_text, audio_path):
    try:
        speech_rate = 0
        pause_rate = 0
        pronunciation_accuracy = 0
        real_and_transcribed_words_ipa = []
        mfcc_features = None
        
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
                model = whispert.load_model("avintech/whisper-small-chinese", device="cuda")
            case "ms":
                model = whispert.load_model("avintech/whisper-small-malay", device="cuda")
            case "ta":
                model = whispert.load_model("avintech/whisper-small-tamil", device="cuda")
            case _:
                model = whispert.load_model("base")

        result = whispert.transcribe(model, audio_np, language=language, detect_disfluencies=True)
        recorded_audio_text = result["text"]
        words_list = []
        pause_list = []
        for segment in result["segments"]:
            for word in segment["words"]:
                if word['text'] != "[*]":  
                    words_list.append(word)
                else:
                    pause_list.append(word)

        real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices = matchWords(language, provided_text, recorded_audio_text)
        pronunciation_accuracy, current_words_pronunciation_accuracy = getPronunciationAccuracy(real_and_transcribed_words_ipa)
        total_duration = result["segments"][-1]["end"]
        speech_rate = calculate_speech_rate(words_list, total_duration)
        pause_rate = get_pause_rate(pause_list, total_duration)
        #mfcc = get_mfcc(audio_data, sample_rate, result['segments'])
        mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20).T, axis=0)
        print(type(mfcc))
    except Exception as ex:
        print(f"Error in load_audio function: {ex}")
    
    finally:
        data = {
            'real_and_transcribed_words': real_and_transcribed_words,
            'wav_file_path': audio_path,
            'speech_rate': speech_rate,
            'pause_rate': pause_rate,
            'pronunciation_accuracy': pronunciation_accuracy,
            'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa,
            'mfcc': mfcc,
            # 'mean_pitch': mean_pitch,
            # 'pitch_range': pitch_range,
            # 'std_pitch': std_pitch,
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
    
    def extract_mfcc(y, sr, start, end, n_mfcc=13):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        return mfccs

    def pad_or_truncate_mfccs(mfccs_list, max_length):
        padded_mfccs = []
        for mfcc in mfccs_list:
            if mfcc.shape[1] > max_length:
                mfcc = mfcc[:, :max_length]
            else:
                mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
            padded_mfccs.append(mfcc)
        return np.array(padded_mfccs)
    
    try:
        all_mfccs = []
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            mfccs = extract_mfcc(y, sr, start_time, end_time)
            all_mfccs.append(mfccs)
        
        max_length = 100
        padded_mfccs = pad_or_truncate_mfccs(all_mfccs, max_length)
        print(f"Padded MFCCs shape: {padded_mfccs.shape}")

        scaler = StandardScaler()
        normalized_mfccs = scaler.fit_transform(padded_mfccs.reshape(-1, padded_mfccs.shape[-1])).reshape(padded_mfccs.shape)

        print(normalized_mfccs.dtype)
        print(f"Normalized MFCCs shape: {normalized_mfccs.shape}")
    except Exception as ex:
        print(f"Error in get_mfcc function: {ex}")
        normalized_mfccs = None

    return normalized_mfccs