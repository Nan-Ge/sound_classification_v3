import librosa
import librosa.display
import matplotlib.pyplot as plt

from utils.spec_visualization import spec_calc
from utils.audio_data_load import arg_list, load_npy



def self_vib_cancellation(mix_audio_file, pure_audio_file):
    pass


if __name__ == '__main__':
    feat_type = 'fbank'
    deno_method = 'pywt'  # (skimage-Visu, skimage-Bayes, pywt)
    kargs = arg_list(fs=48000, n_fft=256, win_len=256, hop_len=64, n_mels=40)

    max_len = 6000
    interval = [0, 1]
    y_axis_type = 'linear'

    mix_audio = load_npy(audio_filepath='mix.npy', max_len=max_len, interval=interval, max_norm=False)[0]
    pure_audio = load_npy(audio_filepath='pure.npy', max_len=max_len, interval=interval, max_norm=False)[0]
    diff_audio = mix_audio - pure_audio

    fig, ax = plt.subplots(nrows=1, ncols=3)

    spec_1 = spec_calc(audio_data=mix_audio, kargs=kargs, feat_type=feat_type)
    librosa.display.specshow(spec_1, y_axis=y_axis_type, x_axis='time', sr=kargs.fs, ax=ax[0])
    ax[0].set(title='Mix-Audio')
    ax[0].get_xaxis().set_visible(False)
    ax[0].label_outer()

    spec_2 = spec_calc(audio_data=pure_audio, kargs=kargs, feat_type=feat_type)
    librosa.display.specshow(spec_2, y_axis=y_axis_type, x_axis='time', sr=kargs.fs, ax=ax[1])
    ax[1].set(title='Pure-Audio')
    ax[1].get_xaxis().set_visible(False)
    ax[1].label_outer()

    # spec_3 = stft_visualization(sound_data=diff_audio, sampling_rate=fs)
    spec_3 = spec_1 - spec_2
    img = librosa.display.specshow(spec_3, y_axis=y_axis_type, x_axis='time', sr=kargs.fs, ax=ax[2])
    ax[2].set(title='Diff-Audio')
    ax[2].get_xaxis().set_visible(False)
    ax[2].label_outer()

    fig.colorbar(img, ax=ax, format="%+2.f dB")
    fig.suptitle("Mix-Pure Audio")
    plt.savefig("result_2.png")
    plt.cla()




