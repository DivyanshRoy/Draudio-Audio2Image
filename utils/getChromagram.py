import librosa, librosa.display

def getChromagram(filename):
    x, sr = librosa.load(filename)
    fmin = librosa.midi_to_hz(36)
    hop_length = 512
    C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    return chromagram
