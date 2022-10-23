from playsound import playsound

def playSound(instrument, note):
    if instrument == "drum":
        playRandom()
    elif instrument == "piano":
        playPiano(note, 4)
    elif instrument == "hat":
        playHighHat()
    else:
        playRandom()

def playPiano(note: str, scale: int):
    playsound("./piano-mp3/" + note + str(scale) + ".mp3")

def playDrum():
    print("playing drum")

def playHighHat():
    playPiano("A", 7)

def playRandom():
    print("playing the somehow random noise")