import librosa as lb
import sounddevice as sd
import numpy as np
import scipy.signal as signal

import numpy as np
import matplotlib.pyplot as plt

class AudioTools:

    def __init__(self):
        # constants
        self.fs = 44100
        self.duration =  0.25 # Duration of each recording chunk
        self.freq_threshold = 0.1  # Magnitude freq_threshold for valid frequencies
        self.silence_threshold_db = -60

    def is_silent(self, audio):
        ''' Returns True if no note is detected (silence/noise), False otherwise.'''

        if np.max(np.abs(audio)) < 1e-5:  # Handle all-zero audio
            return True

        rms = lb.feature.rms(y=audio)[0]
        peak_rms_db = lb.amplitude_to_db(rms.max(), ref=1.0)

        return peak_rms_db < self.silence_threshold_db


    def note_detect(self, sound):
        ''' Simple function that returns the note being played from an np.array `sound` '''
        if self.is_silent(sound):
            return None

        file_length= len(sound)

        # Fourier transformation from numpy module
        fourier = np.fft.fft(sound)

        # plt.plot(fourier)
        plt.show()

        fourier = np.absolute(fourier)
        max_indx=np.argmax(fourier)
 
        freq=(max_indx*self.fs)/(file_length) # Formula to convert index into sound frequency

        print(f"Freq is {freq}")

        return lb.hz_to_note(freq)

    def record(self):
        ''' Records `duration` seconds of sound and returns an np array of that sound'''
        try:
            myarray = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1, dtype=np.float32)
            sd.wait()
            return myarray.flatten()  # Flatten to 1D array
        except Exception as e:
            print(f"Recording failed: {e}")
            return None

    def play(self, sound):
            if sound is not None:
                sd.play(sound, samplerate= self.fs)
                sd.wait()

    def process_audio_fft(self, audio_data):
        ''' uses the fast fourrier transform to proccess the audio '''
        if audio_data is None or len(audio_data) == 0:
            print("No audio data to process.")
            return None

        # Normalize the audio signal for better results
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Apply a Hanning window to reduce noise
        window = signal.windows.hann(len(audio_data)) 
        windowed_audio = audio_data * window

        # Compute Fourrier transform
        fft_result = np.fft.rfft(windowed_audio)
        frequencies = np.fft.rfftfreq(len(windowed_audio), d=1/self.fs)
        magnitude = np.abs(fft_result)

        # Find the peak frequency
        peak_index = np.argmax(magnitude)
        if peak_index >= len(frequencies):
            print("Error: Peak index out of bounds")
            return None

        peak_freq = frequencies[peak_index]
        print(f"Detected Frequency: {peak_freq:.2f} Hz")

        return peak_freq

    def _note_detect(self, audio_data):
        ''' More complicated legacy note_detect algorithm. The simple algorithm works better  '''
        # uses librosa to process audio
        if audio_data is None or len(audio_data) == 0:
            print("No audio data to process.")
            return None

        # Normalize the audio signal
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Use librosa's pitch detection to find the fundamental frequency
        try:
            # Detect pitch using librosa's piptrack
            pitches, magnitudes = lb.core.piptrack(
                y=audio_data,
                sr=self.fs,
                fmin=80,  # Minimum frequency (e.g., lower bound for musical notes)
                fmax=1000  # Maximum frequency (e.g., upper bound for musical notes)
            )

            # Get the pitch with the highest magnitude
            pitch_index = magnitudes.argmax()
            pitch_freq = pitches[pitch_index // magnitudes.shape[1], pitch_index % magnitudes.shape[1]]

            # Ignore frequencies with very low magnitudes (likely noise)
            if magnitudes.max() < self.freq_threshold:
                print("No significant frequency detected.")
                return None


            # Convert frequency to note
            note = lb.hz_to_note(pitch_freq)

            return note
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None


    
