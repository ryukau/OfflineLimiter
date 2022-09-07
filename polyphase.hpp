// Copyright (C) 2022  Takamitsu Endo
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

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

  void setFir(const double *source, size_t start, size_t end)
  {
    std::copy(source + start, source + end, coefficient);

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

template<typename Sample, typename Fir> class FirUpSampler {
  std::array<OverlapSaveConvolver, Fir::upfold> convolvers{};

public:
  std::array<Sample, Fir::upfold> output;

  FirUpSampler()
  {
    for (size_t idx = 0; idx < Fir::upfold; ++idx) {
      convolvers[idx].init(Fir::bufferSize, Fir::intDelay);
      convolvers[idx].setFir(&(Fir::coefficient[idx][0]), 0, Fir::bufferSize);
      convolvers[idx].reset();
    }
  }

  void reset()
  {
    for (auto &cv : convolvers) cv.reset();
  }

  void process(Sample input)
  {
    for (size_t idx = 0; idx < Fir::upfold; ++idx) {
      output[idx] = convolvers[idx].process(input);
    }
  }
};

template<typename Sample, typename Fir> class FirDownSampler {
  std::array<OverlapSaveConvolver, Fir::upfold> convolvers{};

public:
  FirDownSampler()
  {
    for (size_t idx = 0; idx < Fir::upfold; ++idx) {
      convolvers[idx].init(Fir::bufferSize, Fir::intDelay);
      convolvers[idx].setFir(&(Fir::coefficient[idx][0]), 0, Fir::bufferSize);
      convolvers[idx].reset();
    }
  }

  void reset()
  {
    for (auto &cv : convolvers) cv.reset();
  }

  Sample process(std::array<Sample, Fir::upfold> &input)
  {
    for (size_t idx = 0; idx < Fir::upfold; ++idx) {
      input[idx] = convolvers[idx].process(input[idx]);
    }
    return std::accumulate(input.begin(), input.end(), Sample(0));
  }
};

inline std::vector<double>
getNuttallFir(size_t nTap, double sampleRate, double cutoffHz, bool isHighpass)
{
  const auto nyquist = sampleRate / double(2);
  if (cutoffHz > nyquist) cutoffHz = nyquist;

  bool isEven = (nTap / 2 & 1) == 0;
  size_t end = nTap;
  if (isEven) --end; // Always use odd length FIR.

  std::vector<double> coefficient(nTap);

  auto mid = double(end - 1) / double(2);
  auto cutoff = double(2.0 * std::numbers::pi) * cutoffHz / sampleRate;
  for (size_t idx = 0; idx < end; ++idx) {
    double m = double(idx) - mid;
    double x = cutoff * m;
    coefficient[idx] = x == 0 ? double(1) : std::sin(x) / (x);
  }

  // Apply Nuttall window.
  double tpN = double(2.0 * std::numbers::pi) / double(end - 1);
  for (size_t n = 0; n < end; ++n) {
    auto c0 = double(0.3635819);
    auto c1 = double(0.4891775) * std::cos(tpN * n);
    auto c2 = double(0.1365995) * std::cos(tpN * n * double(2));
    auto c3 = double(0.0106411) * std::cos(tpN * n * double(3));
    coefficient[n] *= c0 - c1 + c2 - c3;
  }

  // Normalize to fix FIR scaling.
  double sum = std::accumulate(coefficient.begin(), coefficient.end(), double(0));
  for (size_t idx = 0; idx < end; ++idx) coefficient[idx] /= sum;

  if (isHighpass) {
    for (size_t idx = 0; idx < end; ++idx) coefficient[idx] = -coefficient[idx];
    coefficient[size_t(mid)] += double(1);
  }

  return coefficient;
}
