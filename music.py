from pypiano import Piano
from mingus.containers import Note

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
    p = Piano()
    p.play(note + '-' + str(scale))

def playDrum():
    print("playing drum")

def playHighHat():
    print("playing piano")

    playPiano("A", 7)

def playRandom():
    print("playing the somehow random noise")