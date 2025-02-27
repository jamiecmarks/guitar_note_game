from audiotools import AudioTools

def main():
    at = AudioTools()
    while True:
        recorded_audio = at.record()
        if recorded_audio is not None:
            print(f"Recorded audio shape: {recorded_audio.shape}")
            note = at.note_detect(recorded_audio)
            print(f"Note: {note}")

if __name__ == "__main__":
    main()
