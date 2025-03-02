import random
import time
from audiotools import AudioTools
import re 

class GameInstance:
    def __init__(self):
        letters = list("abcdefg".upper())
        self.notes = [letter + "#" for letter in letters if letter not in "BE"]  + letters # All the valid music notes
        # self.curr_note = None
        self.score = 0
        self.duration_mins = 2 # how long the game should go on for in minutes
        self.at = AudioTools()

    def choose_note(self) -> str:
        return random.choice(self.notes)

    def process_note(self, note):
        num = int(re.findall(r"[0-9]", note)[0])
        only_note = note[0:-1] # everything but the end number
        
        if len(only_note) > 1:
            only_note =  only_note[0] + "#" # avoid the weird ascii # that librosa uses

        return only_note, num


    def play(self):
        start = time.time() # start time
        length = 0
        curr_note = None

        while length <= self.duration_mins * 60:
            if curr_note == None: # Note played or not yet assigned
                curr_note = self.choose_note()
                print(curr_note)

            curr = time.time()
            length = curr - start

            audio_data = self.at.record()

            full_note = self.at.note_detect(audio_data)

            if not full_note: # no note detected
                continue

            played_note, num = self.process_note(full_note)

            # print(curr_note, played_note)
            # print(played_note)


            if curr_note == played_note:
                curr_note = None
                self.score += 1

                print(f"Nice! Score is {self.score}")
