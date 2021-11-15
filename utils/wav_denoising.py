import pywt
from skimage.restoration import denoise_wavelet


def denoising(sound_data, method='skimage-Visu'):
    wavelet_func = 'sym8'
    wavelet_decomposer = pywt.Wavelet(wavelet_func)
    max_dec_level = pywt.dwt_max_level(len(sound_data), wavelet_decomposer.dec_len)

    if method == 'skimage-Visu':
        # sound_data = sound_data / max(sound_data)  # Normalization

        denoise_sound = denoise_wavelet(sound_data,
                                        method='VisuShrink',
                                        mode='soft',
                                        wavelet_levels=max_dec_level,
                                        wavelet=wavelet_func,
                                        rescale_sigma='True')

        return denoise_sound

    elif method == 'skimage-Bayes':
        # sound_data = sound_data / max(sound_data)  # Normalization

        denoise_sound = denoise_wavelet(sound_data,
                                        method='BayesShrink',
                                        mode='soft',
                                        wavelet_levels=max_dec_level,
                                        wavelet=wavelet_func,
                                        rescale_sigma='True')

        return denoise_sound

    elif method == 'pywt':
        threshold = 0.05
        wavelet_coeffs = pywt.wavedec(sound_data, wavelet_func, level=max_dec_level)  # 小波分解
        for i in range(1, len(wavelet_coeffs)):
            wavelet_coeffs[i] = pywt.threshold(wavelet_coeffs[i], threshold * max(wavelet_coeffs[i]))  # 阈值化
        sound_rec = pywt.waverec(wavelet_coeffs, wavelet_func)  # 小波重构

        return sound_rec
