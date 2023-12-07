import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile
import subprocess


def testExpSine(filename, samplerate, freq=100, minAmp=1, maxAmp=1e5):
    phase = np.empty(samplerate)
    ph = 0
    for idx in range(len(phase)):
        phase[idx] = ph
        ph += freq / samplerate
        ph -= np.floor(ph)
    sig = np.sin(2 * np.pi * phase) * np.geomspace(minAmp, maxAmp, samplerate)
    soundfile.write(filename, sig, samplerate, subtype="FLOAT")
    return sig


def normalizeAmplitude(sig, samplerate, windowLengthSeconds=10, targetAmpDecibel=-10):
    length = int(samplerate * windowLengthSeconds)
    window = np.full(length, 1 / length)
    maxSquaredAvg = np.max(signal.convolve(sig * sig, window, mode="same"))
    if maxSquaredAvg <= 0:
        return sig  # Signal is silence.
    peakRms = np.sqrt(maxSquaredAvg)
    targetAmp = 10 ** (targetAmpDecibel / 20)
    return sig * (targetAmp / peakRms)


def testNormalize(source_filename, plot=False):
    samplerate = 48000
    src = testExpSine(source_filename, samplerate)
    ref = normalizeAmplitude(src, samplerate)

    # `--threshold` is set to +140 dB (== 1e7) to bypass limiter.
    polyphase_filename = "polyphase_" + source_filename
    subprocess.run(
        [
            "../build/Release/offlinelimiter",
            "--prompt",
            "yes",
            "--trim",
            "--input",
            source_filename,
            "--output",
            polyphase_filename,
            "--threshold",
            "140",
            "--normalize",
            "-10",
        ]
    )

    fft_filename = "fft_" + source_filename
    subprocess.run(
        [
            "../build/Release/offlinelimiter",
            "--prompt",
            "yes",
            "--upsample",
            "16",
            "--input",
            source_filename,
            "--output",
            fft_filename,
            "--threshold",
            "140",
            "--normalize",
            "-10",
        ]
    )

    if not plot:
        return
    # plt.plot(src, label="source")
    plt.plot(ref, color="black", alpha=0.1, lw=8, label="reference")
    data, fs = soundfile.read(polyphase_filename)
    plt.plot(data, color="red", alpha=0.5, label="polyphase")
    data, fs = soundfile.read(fft_filename)
    plt.plot(data, color="blue", alpha=0.5, label="fft")
    plt.grid()
    plt.legend()
    # plt.savefig(f"{output_filename}.png")
    # plt.close()
    plt.show()


if __name__ == "__main__":
    testNormalize("expsine.wav", True)
