
import sounddevice as sd
import numpy as np

fs = 44100

def record():
    duration = 2  # seconds
    
    # Start recording
    myarray = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)

    # Wait for recording to complete
    sd.wait()


    return myarray

def play(sound):
    # Play the recorded sound
    sd.play(sound, samplerate=fs)

    # Ensure playback completes
    sd.wait()

def main():
    while True:
        recorded_audio = record()
        play(recorded_audio)
        print(recorded_audio.shape)

if __name__ == "__main__":
    main()

