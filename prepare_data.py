import eng_to_ipa
print(f"eng_to_ipa: 0.0.2")
import numpy as np
import os
import parselmouth
print(f"parselmouth: {parselmouth.__version__}")
from pydub import AudioSegment
print(f"pydub: 0.25.1")
import scipy.signal
print(f"scipy: {scipy.__version__}")
from string import punctuation
import whisper_timestamped as whispert
print(f"whisper_timestamped: 1.15.3")
import WordMatching
import WordMetrics
import zaf

def load_audio(provided_text,audio_path):
    try:
        speech_rate = 0
        pause_rate = 0
        pronunciation_accuracy = 0
        
        #convert mp3 to wav
        root, ext = os.path.splitext(audio_path)
        if ext == ".mp3":
            sound = AudioSegment.from_mp3(audio_path)
            sound.export(audio_path.replace('mp3','wav'), format="wav")
            audio_path = audio_path.replace('mp3','wav')
        
        #get transcription
        audio = whispert.load_audio(audio_path)
        model = whispert.load_model("base")
        result = whispert.transcribe(model, audio, language='en', detect_disfluencies=True)
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
        real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices = matchWords(provided_text, recorded_audio_text)
        print(real_and_transcribed_words_ipa)
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

def matchWords(provided_text, recorded_transcript):
    def convertToPhonem(sentence: str) -> str:
        phonem_representation = eng_to_ipa.convert(sentence)
        phonem_representation = phonem_representation.replace('*','')
        return phonem_representation
    
    words_estimated = recorded_transcript.split()
    words_real = provided_text.split()

    mapped_words, mapped_words_indices = WordMatching.get_best_mapped_words(
        words_estimated, words_real)

    real_and_transcribed_words = []
    real_and_transcribed_words_ipa = []
    for word_idx in range(len(words_real)):
        if word_idx >= len(mapped_words)-1:
            mapped_words.append('-')
        real_and_transcribed_words.append(
            (words_real[word_idx],    mapped_words[word_idx]))
        real_and_transcribed_words_ipa.append((convertToPhonem(words_real[word_idx]),
                                            convertToPhonem(mapped_words[word_idx])))
    return real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices

def getPronunciationAccuracy(real_and_transcribed_words_ipa) -> float:
    def removePunctuation(word: str) -> str:
        return ''.join([char for char in word if char not in punctuation])
    
    total_mismatches = 0.
    number_of_phonemes = 0.
    current_words_pronunciation_accuracy = []
    for pair in real_and_transcribed_words_ipa:
        real_without_punctuation = removePunctuation(pair[0]).lower()
        number_of_word_mismatches = WordMetrics.edit_distance_python(
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
    audio_signal, sampling_frequency = zaf.wavread(audio_path)

    # Check if audio_signal is multi-channel and average it over its channels if so
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Set the parameters for the Fourier analysis
    window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
    window_function = scipy.signal.hamming(window_length, sym=False)
    step_length = int(window_length / 2)

    # Compute the mel filterbank
    number_mels = 40
    mel_filterbank = zaf.melfilterbank(sampling_frequency, window_length, number_mels)

    # Compute the MFCCs using the filterbank
    number_coefficients = 20
    audio_mfcc = zaf.mfcc(audio_signal, window_function, step_length, mel_filterbank, number_coefficients)

    # Assuming 'mfcc' is your array of MFCC features
    mfcc_mean = np.mean(audio_mfcc, axis=1)
    mfcc_std = np.std(audio_mfcc, axis=1)
    mfcc_min = np.min(audio_mfcc, axis=1)
    mfcc_max = np.max(audio_mfcc, axis=1)
    
    return mfcc_mean, mfcc_std, mfcc_min, mfcc_max