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

Apply offline limiter to `impulse.wav`.

```
../build/Release/offlinelimiter --verbose --prompt yes --input impulse.wav
```

Check `limiterd_impulse.wav` with some audio editor. If you don't have one, Audacity may be used.
