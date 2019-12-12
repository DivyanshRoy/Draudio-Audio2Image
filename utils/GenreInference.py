import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
import imageio
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def getGenre(filename):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
            self.pool = nn.MaxPool2d(4, 4)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
            self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
            self.fc1 = nn.Linear(32 * 10 * 8, 64)
            self.dropout = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            n = x.size()[0]
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(n, 32 * 10 * 8)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    def inference():
        img = np.array(imageio.imread("inference.png"))
        img = img[:, :, :3]
        img = np.reshape(img, newshape=[1, img.shape[0], img.shape[1], img.shape[2]])
        img = np.transpose(img, axes=(0, 3, 1, 2))

        net = Net()
        net.eval()
        PATH = 'model/musicGenrePredictionModel.pth'
        net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        with torch.no_grad():
            images = torch.from_numpy(img)
            images = images.float()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()[0], outputs.data.numpy()


    sound = AudioSegment.from_mp3(filename)
    sound = sound.set_channels(1)
    sound.export("converted.wav", format="wav")
    samplingFrequency, signalData = wavfile.read('converted.wav')
    signalDataMain = np.array(signalData)

    predCounts = np.zeros(shape=[10], dtype=np.int32)

    for i in range(0, signalDataMain.shape[0], samplingFrequency*30):
        signalData = signalDataMain[i:i+samplingFrequency*30]
        signalData = np.minimum(signalData, 5000)
        signalData = np.maximum(signalData, -5000)
        plt.specgram(signalData, Fs=samplingFrequency, cmap='jet')
        plt.axis('off')
        plt.savefig("spec.png")#, bbox_inches='tight')
        img = imageio.imread("spec.png")
        img = Image.fromarray(img)
        img = img.resize((678, 512), Image.ANTIALIAS)
        img = np.array(img)
        imageio.imsave("inference.png", img)
        prediction, out = inference()
        predCounts[prediction] += 1
    finalPrediction = np.argmax(predCounts)
    classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    genre = classes[finalPrediction]
    return genre
