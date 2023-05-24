# In[1]:

#cross_entropy_model.py
from abc import abstractmethod
from torch import nn
import librosa
import numpy as np
import python_speech_features as psf
import torch
import torch.nn.functional as F
import os

# from fbank_net.model_training.base_model import FBankNet

class FBankResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.network(x)
        out = out + x
        out = self.relu(out)
        return out


class FBankNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=32, out_channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=128, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=256, out_channels=256, kernel_size=3),
            nn.AvgPool2d(kernel_size=4)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(256, 250)
        )

    @abstractmethod
    def forward(self, *input_):
        raise NotImplementedError('Call one of the subclasses of this class')
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=32, out_channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=128, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=256, out_channels=256, kernel_size=3),
            nn.AvgPool2d(kernel_size=4)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(256, 250)
        )

    @abstractmethod
    def forward(self, *input_):
        raise NotImplementedError('Call one of the subclasses of this class')

class FBankCrossEntropyNet(FBankNet):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_layer = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x):
        n = x.shape[0]
        out = self.network(x)
        out = out.reshape(n, -1)
        out = self.linear_layer(out)
        return out

    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val


# In[2]:
#preprocessing.py demo folder
def get_fbanks(audio_file):
    
    def normalize_frames(signal, epsilon=1e-12):
        return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in signal])

    y, sr = librosa.load(audio_file, sr=16000)
    assert sr == 16000

    trim_len = int(0.25 * sr)
    if y.shape[0] < 1 * sr:
        # if less than 1 seconds, don't use that audio
        return None

    y = y[trim_len:-trim_len]

    # frame width of 25 ms with a stride of 15 ms. This will have an overlap of 10s
    filter_banks, energies = psf.fbank(y, samplerate=sr, nfilt=64, winlen=0.025, winstep=0.01)
    filter_banks = normalize_frames(signal=filter_banks)

    filter_banks = filter_banks.reshape((filter_banks.shape[0], 64, 1))
    return filter_banks


def extract_fbanks(path):
    fbanks = get_fbanks(path)
    num_frames = fbanks.shape[0]

    # sample sets of 64 frames each

    numpy_arrays = []
    start = 0
    while start < num_frames + 64:
        slice_ = fbanks[start:start + 64]
        if slice_ is not None and slice_.shape[0] == 64:
            assert slice_.shape[0] == 64
            assert slice_.shape[1] == 64
            assert slice_.shape[2] == 1

            slice_ = np.moveaxis(slice_, 2, 0)
            slice_ = slice_.reshape((1, 1, 64, 64))
            numpy_arrays.append(slice_)
        start = start + 64

    print('num samples extracted: {}'.format(len(numpy_arrays)))
    return np.concatenate(numpy_arrays, axis=0)


# In[3]:



def get_cosine_distance(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    return (1 - F.cosine_similarity(a, b)).numpy()

def get_embeddings(x):
    x = torch.from_numpy(x)
    with torch.no_grad():
        embeddings = model_instance(x)
    return embeddings.numpy()

pt='triplet_loss_trained_model.pth'
model_instance = FBankCrossEntropyNet()
model_instance.load_state_dict(torch.load(pt, map_location=lambda storage, loc: storage))
model_instance = model_instance.double()
model_instance.eval()

# In[ ]:
#when user registers convert .flac to .npy as stored embeddings of thet user.
DATA_DIR="embeddings/"
class VoiceAuth:
    def register_user(self, filename):
        fbanks = extract_fbanks(filename)
        filename_flac = filename.split('/')[1]
        person_name = filename_flac.split('.')[0]
        embeddings = get_embeddings(fbanks)
        print('shape of embeddings: {}'.format(embeddings.shape), flush=True)
        mean_embeddings = np.mean(embeddings, axis=0)
        np.save(DATA_DIR + person_name + '.npy', mean_embeddings)

    THRESHOLD = 0.45 

    def loginSuccess(self, filename):
        filename_flac = filename.split('/')[1]
        person_name = filename_flac.split('.')[0]
        fbanks = extract_fbanks(filename)
        embeddings = get_embeddings(fbanks)

        npyNm = DATA_DIR + person_name + '.npy'

        if os.path.isfile(npyNm) == False:
            return False


        print("npy_ nm : "+npyNm)
        stored_embeddings = np.load(npyNm)
        stored_embeddings = stored_embeddings.reshape((1, -1))

        distances = get_cosine_distance(embeddings, stored_embeddings)
        print('mean distances', np.mean(distances), flush=True)
        positives = distances < 0.45
        positives_mean = np.mean(positives)
        print('positives mean: {}'.format(positives_mean), flush=True)
        if positives_mean >= .60:
            print('success')
            return True
        else:
            print('failure')
            return False




