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

        # #compute pitch
        # mean_pitch , pitch_range, std_pitch = get_pitch(audio_path)
        
        # #compute MFCC
        # mfcc_mean, mfcc_std, mfcc_min, mfcc_max = get_mfcc(audio_path)
        # # Combine these into a single feature vector
        # mfcc = np.concatenate([mfcc_mean, mfcc_std, mfcc_min, mfcc_max])
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
            #'mfcc': mfcc,
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

def get_mfcc(audio_path):
    audio_signal, sampling_frequency = wavread(audio_path)

    # Check if audio_signal is multi-channel and average it over its channels if so
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Set the parameters for the Fourier analysis
    window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
    window_function = scipy.signal.hamming(window_length, sym=False)
    step_length = int(window_length / 2)

    # Compute the mel filterbank
    number_mels = 40
    mel_filterbank = melfilterbank(sampling_frequency, window_length, number_mels)

    # Compute the MFCCs using the filterbank
    number_coefficients = 20
    audio_mfcc = mfcc(audio_signal, window_function, step_length, mel_filterbank, number_coefficients)

    # Assuming 'mfcc' is your array of MFCC features
    mfcc_mean = np.mean(audio_mfcc, axis=1)
    mfcc_std = np.std(audio_mfcc, axis=1)
    mfcc_min = np.min(audio_mfcc, axis=1)
    mfcc_max = np.max(audio_mfcc, axis=1)
    
    return mfcc_mean, mfcc_std, mfcc_min, mfcc_max