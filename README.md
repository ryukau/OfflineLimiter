# Offline Limiter
Offline limiter is an almost true peak limiter. "Almost" here means that accuracy depends on up-sampling ratio.

2 different multirate technique is used. One is FIR polyphase, and other is using FFT. Default is FIR polyphase because FFT up-sampling requires large amount of memory.

## Build
Requires C++ compiler with C++20 support.

Dependencies are following.

- Boost::program_options
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

might do the trick with g++ and clang++.

After modification, following command might work. On macOS, it might be better to add `-GXcode` option at line 🤔.

```bash
cd OfflineLimiter
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release # 🤔
cmake --build build --config release
```

## Usage
Detail of limiter parameter is written in BasicLimiter manual linked below. The limiter algorithm is the same. Only multirate processing part is changed.

- [BasicLimiter User Manual](https://ryukau.github.io/VSTPlugins/manual/BasicLimiter/BasicLimiter_en.html)

Below is list of command line options.

```
  -h [ --help ]                         Show this message.
  -v [ --verbose ]                      Show processing status.
  -p [ --prompt ] arg                   Answer and skip prompt when value is
                                        set to "yes" or "no". Otherwise, prompt
                                        will show up.
  -m [ --memory ] arg (=1)              Memory warning threshold in GiB. When
                                        estimated memory allocation exceeds
                                        this value, prompt will show up.
  -i [ --input ] arg                    Input audio file path.
  -o [ --output ] arg                   Output audio file path.
  --precise                             When specified, FFT up-sampling is used
                                        instead of FIR polyphase up-sampling.
                                        FFT up-sampling requires large amount
                                        of memory that is multiple of input
                                        file size and up-sampling ratio.
  -a [ --attack ] arg (=0.0013333333333333333)
                                        Attack time in seconds.
  -s [ --sustain ] arg (=0.0013333333333333333)
                                        Sustain time in seconds.
  -r [ --release ] arg (=0)             Release time in seconds.
  -t [ --threshold ] arg (=-0.10000000000000001)
                                        Limiter threshold in decibel.
  -g [ --gate ] arg (=-inf)             Gate threshold in decibel.
  -l [ --link ] arg (=0.5)              Stereo or multi-channel link amount in
                                        [0.0, 1.0]. 0.0 is no link, and 1.0 is
                                        full link.
  -u [ --upsample ] arg (=16)           Up-sampling ratio.
```

Below is an example invocation in PowerShell.

```ps1
offlinelimiter.exe `
  --verbose `
  --prompt yes `
  --input .\data\oracleengine.wav `
  --release 0.02
```

## `--precise` Mode
### True Peak Reconstruction
True peak here means that absolute maximum of sinc interpolated signal.

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

And I define true peak as following.

$$
\mathtt{truepeak} = \max(|x(t)|), \enspace \text{for all} \enspace t \enspace \text{in} \enspace \mathbb{R}.
$$

As you can see, sinc interpolation requires infinite length convolution. It can't be computed in real-time or in real-life. However, we can approximate true peak by using up-sampling with discrete fourier transform (DFT). I found this from experiment, but I guess there are some book explains the theory behind it. The idea is that ideal lowpass equation matches to DFT equation. Also the use of DFT is the reason this limiter runs offline, not in real-time. It has to know all the input beforehand.

The accuracy of true peak reconstruction depends on how fine grained the up-sampling is. I'm not good at math, but it might be written as following.

$$
\mathtt{truepeak} \approx \max(|x(\tau)|)
, \enspace \forall
\tau \in \left\{
  a + b \mid
\forall a \in \{0, 1, \dots, N-1\},\enspace
\forall b \in \left\{ \frac{0}{L}, \frac{1}{L}, \dots, \frac{L-1}{L} \right\}
\right\}.
$$

- $\tau$ is up-sampled discrete time in samples.
- $N$ is number of input samples.
- $L$ is up-sampling ratio.

The idea is that if we increase $L$ to $+\infty$, then $\tau$ becomes almost same as $t$. That's what I'm calling almost true peak. Note that this is not math proof, but conveying idea using math equation. I'm not sure if $t$ and $\tau$ becomes equal in case of $L \to +\infty$. Also the above approximation is assuming that $x(t)$ is $0$ where $t < 0$, or $t > N-1$.

### Limiter Algorithm
Limiter algorithm is the same one used in BasicLimiter.

- [BasicLimiter User Manual](https://ryukau.github.io/VSTPlugins/manual/BasicLimiter/BasicLimiter_en.html)
- [Source Code](https://github.com/ryukau/VSTPlugins/blob/master/BasicLimiter/source/dsp/limiter.hpp)

Details are written in following link.

- [リミッタの実装](https://ryukau.github.io/filter_notes/limiter/limiter.html)

## FIR Polyphase
FIR polyphase specification:

- 1023 taps high elimination lowpass.
- 1023 taps * 8 phase up-sampler.
- 511 taps * 8 phase down-sampler.

Cutoff frequency of all 3 filters is 23500 Hz at 48000 Hz sampling rate.

Comments in the codes in `fir` directory contains Python3 code that is used to design filter.

## License
GPLv2+ due to linking to FFTW3.
