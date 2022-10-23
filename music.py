from playsound import playsound

def playSound(instrument, note):
    if note == "":
        note = "D"
    if instrument == "drum":
        playRandom()
    elif instrument == "piano":
        playPiano(note, 4)
    elif instrument == "hat":
        playPiano(note, 7)
    else:
        playRandom()

def playPiano(note: str, scale: int):
    playsound("./piano-mp3/" + note + str(scale) + ".mp3")

def playDrum():
    print("playing drum")

def playRandom():
    print("playing the somehow random noise")