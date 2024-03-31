import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import soundfile
import subprocess


def testImpulse(filename, samplerate):
    sig = np.zeros(samplerate)
    sig[0] = 1
    sig[-1] = 1
    soundfile.write(filename, sig, samplerate)
    return sig


def generateNoise(filename, samplerate, seed=98276521):
    rng = np.random.default_rng(seed)
    sig = 2.0 * rng.binomial(1, 0.5, samplerate) - 1
    soundfile.write(filename, sig, samplerate)
    return sig


def generateStepSignal(filename, samplerate):
    sig = np.ones(samplerate)
    soundfile.write(filename, sig, samplerate)
    return sig


def generateNyquistSignal(filename, samplerate):
    sig = np.empty(samplerate)
    sig[0::2] = 1
    sig[1::2] = -1
    soundfile.write(filename, sig, samplerate)
    return sig


def testPolyPhaseLatency(source_filename, plot=False):
    src = testImpulse(source_filename, 48000)

    output_filename = "polyphase_" + source_filename
    subprocess.run(
        [
            "../build/Release/offlinelimiter",
            "--prompt",
            "yes",
            "--input",
            source_filename,
            "--output",
            output_filename,
            "--trim",
        ]
    )

    if not plot:
        return
    plt.plot(src, label="source")
    data, fs = soundfile.read(output_filename)
    plt.plot(data, label="output")
    plt.grid()
    plt.legend()
    # plt.savefig(f"{output_filename}.png")
    # plt.close()
    plt.show()


def testPolyPhaseLimiting(source_filename, plot=False):
    output_filename = "polyphase_" + source_filename
    subprocess.run(
        [
            "../build/Release/offlinelimiter",
            "--prompt",
            "yes",
            "--input",
            source_filename,
            "--output",
            output_filename,
        ]
    )

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
            output_filename,
        ]
    )

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
            output_filename,
            "--threshold",
            "-0.1",
            "--normalize",
            "-10",
        ]
    )

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

    # testPolyPhaseLatency("impulse.wav", True)

    noise_filename = "binomial_noise.wav"
    generateNoise(noise_filename, samplerate)
    testPolyPhaseLimiting(noise_filename)
    testPreciseLimiting(noise_filename)

    step_filename = "step.wav"
    generateStepSignal(step_filename, samplerate)
    testPolyPhaseLimiting(step_filename)
    testPreciseLimiting(step_filename)

    nyquist_filename = "nyquist.wav"
    generateNyquistSignal(nyquist_filename, samplerate)
    testPolyPhaseLimiting(nyquist_filename)
    testPreciseLimiting(nyquist_filename)

    # testSkipLimiting(samplerate)
