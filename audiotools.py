import librosa as lb
import sounddevice as sd
import numpy as np
import scipy.signal as signal

class AudioTools:

    def __init__(self):
        # constants
        self.fs = 44100
        self.duration =  0.25 # Duration of each recording chunk
        self.threshold = 0.1  # Magnitude threshold for valid frequencies

    def record(self):
        ''' Records `duration` seconds of sound and returns an np array of that sound'''
        try:
            myarray = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1, dtype=np.float32)
            sd.wait()
            return myarray.flatten()  # Flatten to 1D array
        except Exception as e:
            print(f"Recording failed: {e}")
            return None

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

    def process_audio(self, audio_data):
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
            if magnitudes.max() < self.threshold:
                print("No significant frequency detected.")
                return None

            print(f"Detected Frequency: {pitch_freq:.2f} Hz")

            # Convert frequency to note
            note = lb.hz_to_note(pitch_freq)
            print(f"Detected Note: {note}")

            return pitch_freq
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None


    def play(self, sound):
        if sound is not None:
            sd.play(sound, samplerate= self.fs)
            sd.wait()
