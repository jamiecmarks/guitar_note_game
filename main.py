import librosa as lb
from audiotools import AudioTools

def main():
    at = AudioTools()
    while True:
        recorded_audio = at.record()
        if recorded_audio is not None:
            print(f"Recorded audio shape: {recorded_audio.shape}")
            detected_freq = at.process_audio(recorded_audio)
            if detected_freq is not None:
                note = lb.hz_to_note(detected_freq)
                print(f"Note: {note}")
                print(f"Detected Frequency: {detected_freq:.2f} Hz")

if __name__ == "__main__":
    main()
