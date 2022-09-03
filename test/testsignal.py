import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile
import subprocess

def testImpulse(samplerate):
    sig = np.zeros(10000)
    sig[0] = 1
    sig[-1] = 1
    soundfile.write("impulse.wav", sig, samplerate)

def generateNoise(filename, samplerate, seed=98276521):
    rng = np.random.default_rng(seed)
    sig = 2.0 * rng.binomial(1, 0.5, samplerate) - 1
    soundfile.write(filename, sig, samplerate)

def testPolyPhaseLimiting(source_filename, plot=False):
    output_filename = "polyphase_" + source_filename
    subprocess.run([
        "../build/Release/offlinelimiter",
        "--prompt",
        "yes",
        "--input",
        source_filename,
        "--output",
        output_filename,
    ])

    data, fs = soundfile.read(output_filename)
    assert not np.any(np.abs(data) > 1)

    if not plot:
        return
    upsig = signal.resample(data, 32 * len(data))
    plt.plot(upsig)
    plt.grid()
    plt.savefig(f"{output_filename}.png")
    plt.close()

def testPreciseLimiting(source_filename, plot=False):
    output_filename = "fft_" + source_filename
    subprocess.run([
        "../build/Release/offlinelimiter",
        "--prompt",
        "yes",
        "--precise",
        "--input",
        source_filename,
        "--output",
        output_filename,
    ])

    data, fs = soundfile.read(output_filename)
    assert not np.any(np.abs(data) > 1)

    if not plot:
        return
    upsig = signal.resample(data, 32 * len(data))
    plt.plot(upsig)
    plt.grid()
    plt.savefig(f"{output_filename}.png")
    plt.close()

def testSkipLimiting(samplerate, plot=False):
    source_filename = "low_amplitude.wav"
    output_filename = "skipped_" + source_filename

    rng = np.random.default_rng(864864)
    soundfile.write(source_filename, rng.uniform(-0.1, 0.1, samplerate), samplerate)

    subprocess.run([
        "../build/Release/offlinelimiter",
        "--prompt",
        "yes",
        "--upsample",
        "16",
        "--input",
        source_filename,
        "--output",
        output_filename,
    ])

    source, fs = soundfile.read(source_filename)
    output, fs = soundfile.read(output_filename)
    assert np.sum(np.abs(source - output)) == 0

    if not plot:
        return
    plt.plot(source, label="source")
    plt.plot(output, label="output")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_filename}.png")
    plt.close()

if __name__ == "__main__":
    samplerate = 48000

    # testImpulse(samplerate)

    # noise_filename = "binomial_noise.wav"
    # generateNoise(noise_filename, samplerate)
    # testPolyPhaseLimiting(noise_filename)
    # testPreciseLimiting(noise_filename)

    testSkipLimiting(samplerate)
