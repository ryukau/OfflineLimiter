#pragma once

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

class OverlapSaveConvolver {
private:
  static constexpr size_t nBuffer = 2;

  size_t half = 1;
  size_t bufSize = 2;
  size_t spcSize = 1; // spc = spectrum.

  std::array<double *, nBuffer> buf;
  std::complex<double> *spc;
  std::complex<double> *fir;
  double *flt; // filtered.
  double *coefficient;

  std::array<fftw_plan, nBuffer> forwardPlan;
  fftw_plan inversePlan;
  fftw_plan firPlan;

  size_t front = 0;
  std::array<size_t, nBuffer> wptr{};
  size_t rptr = 0;
  size_t offset = 0;

public:
  void init(size_t nTap, size_t delay = 0)
  {
    offset = delay;

    half = nTap;
    bufSize = 2 * half;
    spcSize = nTap + 1;

    for (size_t idx = 0; idx < nBuffer; ++idx) {
      buf[idx] = (double *)fftw_malloc(sizeof(double) * bufSize);
    }
    spc = (std::complex<double> *)fftw_malloc(sizeof(std::complex<double>) * spcSize);
    flt = (double *)fftw_malloc(sizeof(double) * bufSize);

    coefficient = (double *)fftw_malloc(sizeof(double) * bufSize);
    std::fill(coefficient, coefficient + bufSize, double(0));

    fir = (std::complex<double> *)fftw_malloc(sizeof(std::complex<double>) * spcSize);
    std::fill(fir, fir + spcSize, std::complex<double>(0, 0));

    for (size_t idx = 0; idx < nBuffer; ++idx) {
      forwardPlan[idx] = fftw_plan_dft_r2c_1d(
        int(bufSize), buf[idx], reinterpret_cast<fftw_complex *>(spc), FFTW_ESTIMATE);
    }
    inversePlan = fftw_plan_dft_c2r_1d(
      int(bufSize), reinterpret_cast<fftw_complex *>(spc), flt, FFTW_ESTIMATE);
    firPlan = fftw_plan_dft_r2c_1d(
      int(bufSize), coefficient, reinterpret_cast<fftw_complex *>(fir), FFTW_ESTIMATE);
  }

  ~OverlapSaveConvolver()
  {
    for (auto &fp : forwardPlan) fftw_destroy_plan(fp);
    fftw_destroy_plan(inversePlan);
    fftw_destroy_plan(firPlan);

    for (auto &bf : buf) fftw_free(bf);
    fftw_free(spc);
    fftw_free(fir);
    fftw_free(flt);
    fftw_free(coefficient);
  }

  void setFir(std::vector<double> &source, size_t start, size_t end)
  {
    std::copy(source.begin() + start, source.begin() + end, coefficient);

    // FFT scaling.
    for (size_t idx = 0; idx < half; ++idx) coefficient[idx] /= double(bufSize);

    fftw_execute(firPlan);
  }

  void reset()
  {
    wptr[0] = half + offset;
    wptr[1] = offset;
    for (auto &w : wptr) w %= bufSize;
    front = wptr[1] < wptr[0] ? 0 : 1;
    rptr = half + offset % half;

    for (size_t idx = 0; idx < nBuffer; ++idx) {
      std::fill(buf[idx], buf[idx] + bufSize, double(0));
    }
    std::fill(spc, spc + spcSize, std::complex<double>(0, 0));
    std::fill(flt, flt + bufSize, double(0));
  }

  double process(double input)
  {
    buf[0][wptr[0]] = input;
    buf[1][wptr[1]] = input;

    for (auto &w : wptr) {
      if (++w >= bufSize) w = 0;
    }

    if (wptr[front] == 0) {
      fftw_execute(forwardPlan[front]);
      for (size_t i = 0; i < spcSize; ++i) spc[i] *= fir[i];
      fftw_execute(inversePlan);

      front ^= 1;
    }

    if (++rptr >= bufSize) rptr = half;
    return flt[rptr];
  }
};

template<typename Sample, typename Fir> class NaiveConvolver {
private:
  std::array<Sample, Fir::fir.size()> buf{};

public:
  void reset() { buf.fill(Sample(0)); }

  Sample process(Sample input)
  {
    std::rotate(buf.rbegin(), buf.rbegin() + 1, buf.rend());
    buf[0] = input;

    Sample output = 0;
    for (size_t n = 0; n < Fir::fir.size(); ++n) output += buf[n] * Fir::fir[n];
    return output;
  }
};

template<typename Sample, typename Fir> class FirUpSampler {
  std::array<Sample, Fir::bufferSize> buf{};

public:
  std::array<Sample, Fir::upfold> output;

  void reset() { buf.fill(Sample(0)); }

  void process(Sample input)
  {
    std::rotate(buf.rbegin(), buf.rbegin() + 1, buf.rend());
    buf[0] = input;

    std::fill(output.begin(), output.end(), Sample(0));
    for (size_t i = 0; i < Fir::coefficient.size(); ++i) {
      auto &&phase = Fir::coefficient[i];
      for (size_t n = 0; n < phase.size(); ++n) output[i] += buf[n] * phase[n];
    }
  }
};

template<typename Sample, typename Fir> class FirDownSampler {
  std::array<std::array<Sample, Fir::bufferSize>, Fir::upfold> buf{{}};

public:
  void reset() { buf.fill({}); }

  Sample process(const std::array<Sample, Fir::upfold> &input)
  {
    for (size_t i = 0; i < Fir::upfold; ++i) {
      std::rotate(buf[i].rbegin(), buf[i].rbegin() + 1, buf[i].rend());
      buf[i][0] = input[i];
    }

    Sample output = 0;
    for (size_t i = 0; i < Fir::coefficient.size(); ++i) {
      auto &&phase = Fir::coefficient[i];
      for (size_t n = 0; n < phase.size(); ++n) output += buf[i][n] * phase[n];
    }
    return output;
  }
};
