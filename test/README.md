## `testsignal.py`
`testsignal.py` generates test impulse signal to measure the latency of limiter.

The source of latency is following:

- Limiter attack time.
- High elimination filter.
- Up-sampler.
- Down-sampler.

On FFT resampling, limiter attack time is only relevant.

### Usage
Install following dependency.

- [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)
- NumPy

Run `testsignal.py` to generate `impulse.wav`.

```
python testsignal.py
```

### Impulse Response Test
Apply offline limiter to `impulse.wav`.

```
../build/Release/offlinelimiter --verbose --prompt yes --input impulse.wav
```

Check `limiterd_impulse.wav` with some audio editor. If you don't have one, Audacity may be used.

### Limiting Test
`testsignal.py` also tests if true-peak limiting is done correctly or not. See `testPolyPhaseLimiting()` and `testPreciseLimiting` for details.

`*binomial_nosie.wav` are generated for test. The one without prefix is the source. Others are output of offline limiter.
