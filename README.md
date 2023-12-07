# Offline Limiter
Offline Limiter is an almost true-peak limiter. "Almost" here means that accuracy depends on up-sampling ratio.

2 different multirate technique is used. One is FIR polyphase, and other is using FFT. Default is FIR polyphase because FFT up-sampling requires large amount of memory.

It runs offline because it's too heavy.

## Known Bug
When input signal contains sudden amplitude change, and FIR polyphase up-sampling is used, limiting may fail as output exceeds 0 dB. Mitigation is to apply limiter again, or use FFT up-sampling by setting 2 or greater number to `--upsample`.

Possible reason is distortion introduced in limiting process. On up-sampled signal, limiter produces frequency components higher than source Nyquist frequency. However, down-sampler truncates them. This truncation changes the peak in down-sampled signal.

---

When `--fadeout` is non-zero, `--trim` is set, and FIR polyphase up-sampling is used, fade-out is only partially applied to output. Current implementation doesn't account the length of trim.

## Build
Requires C++ compiler with C++20 support.

Dependencies are following.

- Boost::program_options
- Boost::exception
- FFTW3
- libsndfile

### Windows
1. Install Visual Studio with C++20 support.
2. Install CMake.
3. Install vcpkg and dependency listed above.
4. Run following command on PowerShell.

```ps1
cd OfflineLimiter
./build.ps1
```

`offlinelimiter.exe` will be built into `build/Release`.

### Non-Windows
Some modification to top level `CMakeLists.txt` is required.

Changing following part:

```cmake
target_compile_options(${PROJECT_NAME} PRIVATE
  /W4
  /fp:fast
  # /Qvec-report:1
)
```

to:

```cmake
target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall
  -O3
  -ffast-math
)
```

might do the trick for g++ and clang++.

After modification, following command might work. On macOS, it might be better to add `-GXcode` option at line 🤔.

```bash
cd OfflineLimiter
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release # 🤔
cmake --build build --config release
```

## Usage
Output format is fixed to 32-bit float WAV.

Detail of limiter parameter is written in BasicLimiter manual linked below. The limiter algorithm is the same. Only multirate processing part is changed.

- [BasicLimiter User Manual](https://ryukau.github.io/VSTPlugins/manual/BasicLimiter/BasicLimiter_en.html)

Below is the list of command line options.

```
  -h [ --help ]                         Show this message.
  -v [ --verbose ]                      Show processing status.
  -p [ --prompt ] arg                   Answer and skip prompt when value is
                                        set to "yes" or "no". Otherwise, prompt
                                        will show up.
  -i [ --input ] arg                    Input audio file path.
  -o [ --output ] arg                   Output audio file path.
  -u [ --upsample ] arg (=1)            Up-sampling ratio. When set to 1, FIR
                                        polyphase up-sampling is used. When set
                                        to greater than 1, FFT up-sampling is
                                        used. FFT up-sampling requires large
                                        amount of memory that is multiple of
                                        input file size and up-sampling ratio.
                                        If FFT up-sampling is enabled and
                                        up-sampled peak is below threshold,
                                        processing will be skipped. Recommend
                                        to set to 16 or greater for precise
                                        true-peak limiting.
  --trim                                --trim has no effect when --upsample is
                                        set to greater than 1. When specified,
                                        input frame count and output frame
                                        count become the same, by trimming
                                        artifacts introduced by multirate
                                        processing. When not specified, output
                                        signal becomes longer than input
                                        signal. Additional frame count is (158
                                        + attack * samplerate) at front, and
                                        290 at back. Theoretically, trimmed
                                        signal is no longer true-peak limited.
  -m [ --memory ] arg (=1)              Memory warning threshold in GiB. When
                                        estimated memory allocation exceeds
                                        this value, prompt will show up.
  --maxiter arg (=4)                    Maximum iteration count for additional
                                        stage limiting. Sometimes the result of
                                        true-peak limiting still exceeds the
                                        threshold. It's hard to predict the
                                        final sample-peak before donw-sampling.
                                        (If you know the method, please let me
                                        know!) Therefore offlinelimiter applies
                                        extra stage limiting in case of
                                        insufficient limiting. Loop continues
                                        until the final sample-peak becomes
                                        below 0 dB, or iteration count reaches
                                        --maxiter.
  --highpass arg (=0)                   Cutoff frequency of linear phase
                                        highpass filter in Hz. Inactivate when
                                        set to 0. Useful to eliminate direct
                                        current.
  -a [ --attack ] arg (=0.0013333333333333333)
                                        Attack time in seconds.
  -s [ --sustain ] arg (=0.0013333333333333333)
                                        Sustain time in seconds.
  -r [ --release ] arg (=0)             Release time in seconds.
  -t [ --threshold ] arg (=-0.10000000000000001)
                                        Limiter threshold in decibel.
  -g [ --gate ] arg (=-inf)             Gate threshold in decibel. Setting the
                                        value to -inf disables the gate.
  -l [ --link ] arg (=0.5)              Stereo or multi-channel link amount in
                                        [0.0, 1.0]. 0.0 is no link, and 1.0 is
                                        full link.
  -f [ --fadeout ] arg (=0.001)         Fade-out time in seconds. Equal power
                                        curve (or quarter cosine curve) is
                                        used.
  -n [ --normalize ] arg (=-inf)        Target amplitude of normalization in
                                        decibel. Setting the value to +inf,
                                        -inf, or other non-finite value
                                        bypasses normalization. The metering is
                                        10-second moving average. In other
                                        words, this is not sample-peak
                                        normalization, and more close to RMS
                                        normalization.
```

Below is an example invocation in PowerShell.

```ps1
offlinelimiter.exe `
  --verbose `
  --prompt yes `
  --input .\data\oracleengine.wav `
  --release 0.02
```

## Limiter Algorithm
Limiter algorithm is the same one used in BasicLimiter.

- [BasicLimiter User Manual](https://ryukau.github.io/VSTPlugins/manual/BasicLimiter/BasicLimiter_en.html)
- [Source Code](https://github.com/ryukau/VSTPlugins/blob/master/BasicLimiter/source/dsp/limiter.hpp)

Details are written in following link.

- [リミッタの実装](https://ryukau.github.io/filter_notes/limiter/limiter.html)

## True-Peak Reconstruction in `--precise` Mode
True-peak here means that absolute maximum of sinc interpolated signal.

Sinc interpolation is defined as below.

$$
\begin{aligned}
x(t) &= \sum_{n=-\infty}^{\infty} x[n] \mathrm{sinc} \left( t - n \right)\\
\mathrm{sinc}(a) &= \frac{\sin(\pi a)}{\pi a}
\end{aligned}
$$

- $x[n]$ is input signal in discrete time domain.
- $x(t)$ is interpolated signal in continuous time domain.
- $n$ is discrete time in samples.
- $t$ is continuous time in samples. To get time in seconds, divide this value with sampling rate.

And I define true-peak as following.

$$
\mathtt{truepeak} = \max(|x(t)|), \enspace \text{for all} \enspace t \enspace \text{in} \enspace \mathbb{R}.
$$

As you can see, sinc interpolation requires infinite length convolution. It can't be computed in real-time or in real-life. However, we can approximate true-peak by using up-sampling with discrete fourier transform (DFT). I found this from experiment, but I guess there are some books that explain the theory behind it (See reference below). The idea is that ideal lowpass equation matches to DFT equation. Also the use of DFT is the reason this limiter runs offline, not in real-time. It has to know all the input beforehand.

The accuracy of true-peak reconstruction depends on how fine grained the up-sampling is. I'm not good at math, but it might be written as following.

$$
\mathtt{truepeak} \approx \max(|x(\tau)|)
, \enspace \forall
\tau \in \lbrace
  a + b \mid
\forall a \in \lbrace 0, 1, \dots, N-1 \rbrace , \enspace
\forall b \in \lbrace \frac{0}{L}, \frac{1}{L}, \dots, \frac{L-1}{L} \rbrace
\rbrace.
$$

- $\tau$ is up-sampled discrete time in samples.
- $N$ is number of input samples.
- $L$ is up-sampling ratio.

The idea is that if we increase $L$ to $+\infty$, then $\tau$ becomes almost same as $t$. That's what I'm calling almost true-peak. Note that this is not math proof, but conveying idea using math equation. I'm not sure if $t$ and $\tau$ become equal in case of $L \to +\infty$. Also the above approximation is assuming that $x(t)$ is $0$ where $t < 0$, or $t > N-1$.

### Reference
- [Spectral Interpolation](https://ccrma.stanford.edu/~jos/sasp/Spectral_Interpolation.html)
- [Dirichlet kernel - Wikipedia](https://en.wikipedia.org/wiki/Dirichlet_kernel)
- [Dirac comb - Wikipedia](https://en.wikipedia.org/wiki/Dirac_comb)

## FIR Polyphase Spec.
FIR polyphase is used when `--precise` option is not set. This is default because of the better memory efficiency.

Specification:

- 64 taps high elimination lowpass.
- 64 taps * 8 phase up-sampler.
- 64 taps * 8 phase down-sampler.

Because of the high elimination lowpass, the amplitude of frequency components decrease as it approaches to Nyquist frequency.

When `--trim` option is not specified, `158 + attack * samplerate` frames are added to the front, and `290` frames are added to the back of input signal. See usage section for more details.

Comments in the codes in `fir` directory provides Python3 code used to design filter.

## License
GPLv2+
