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

/*

TODO:
- Implement --maxiter to `processMemoryEfficientMode`. Latency must be considered.
- Add fade-in and fade-out.
- Add highpass.
- Maybe add better progress text.
- Maybe add output file format option.
- Maybe add gate.

*/

#include "fir/downsamplercoefficient.hpp"
#include "fir/higheliminatorcoefficient.hpp"
#include "fir/upsamplercoefficient.hpp"
#include "limiter.hpp"
#include "polyphase.hpp"

#include <boost/program_options.hpp>
#include <fftw3.h>
#include <sndfile.h>

#include <chrono>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>

template<typename T> inline T decibelToAmp(T dB) { return std::pow(T(10), dB / T(20)); }
template<typename T> inline T ampToDecibel(T amp) { return 20.0 * std::log10(amp); }

struct FFTW3Buffer {
  double *buf = nullptr;
  std::complex<double> *spc = nullptr;

  size_t bufSize = 0;
  size_t spcSize = 0;

  double peakAmplitude(size_t length)
  {
    if (length > bufSize) {
      std::cerr << "Warning: FFTW3Buffer::peakAmplitude() received length longer than "
                   "buffer size.\n";
      length = bufSize;
    }

    double peak = 0.0f;
    for (size_t i = 0; i < length; ++i) peak = std::max(peak, std::abs(buf[i]));
    return peak;
  }

  void allocate(size_t size)
  {
    bufSize = size;
    spcSize = bufSize / 2 + 1;

    buf = (double *)fftw_malloc(sizeof(double) * bufSize);
    spc = (std::complex<double> *)fftw_malloc(sizeof(std::complex<double>) * spcSize);

    std::fill(buf, buf + bufSize, 0.0);
    std::fill(spc, spc + spcSize, std::complex<double>(0, 0));
  }

  void upSample(size_t fold, size_t frames)
  {
    auto forwardPlan = fftw_plan_dft_r2c_1d(
      int(frames), buf, reinterpret_cast<fftw_complex *>(spc), FFTW_ESTIMATE);
    fftw_execute(forwardPlan);
    fftw_destroy_plan(forwardPlan);

    auto inversePlan = fftw_plan_dft_c2r_1d(
      int(fold * frames), reinterpret_cast<fftw_complex *>(spc), buf, FFTW_ESTIMATE);
    fftw_execute(inversePlan);
    fftw_destroy_plan(inversePlan);

    // FFT normalization.
    for (size_t idx = 0; idx < bufSize; ++idx) buf[idx] /= double(frames);
  }

  void downSample(size_t fold, size_t frames, size_t latency)
  {
    auto forwardPlan = fftw_plan_dft_r2c_1d(
      int(fold * frames), buf + latency, reinterpret_cast<fftw_complex *>(spc),
      FFTW_ESTIMATE);
    fftw_execute(forwardPlan);
    fftw_destroy_plan(forwardPlan);

    auto inversePlan = fftw_plan_dft_c2r_1d(
      int(frames), reinterpret_cast<fftw_complex *>(spc), buf, FFTW_ESTIMATE);
    fftw_execute(inversePlan);
    fftw_destroy_plan(inversePlan);

    // FFT normalization.
    for (size_t idx = 0; idx < frames; ++idx) buf[idx] /= double(bufSize);
  }

  ~FFTW3Buffer()
  {
    if (buf) fftw_free(buf);
    if (spc) fftw_free(spc);

    bufSize = 0;
    spcSize = 0;
  }
};

class SoundFile {
private:
  SNDFILE *file = nullptr;

public:
  SF_INFO info;

  int open(std::string path)
  {
    memset(&info, 0, sizeof(info));

    close();
    file = sf_open(path.c_str(), SFM_READ, &info);
    if (!file) {
      std::cerr << "Error: sf_open failed.\n";
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

  void load(std::vector<FFTW3Buffer> &data)
  {
    if (!file) {
      std::cerr << "Error: SoundFile::load is called before SoundFile::open.\n";
      return;
    }

    sf_count_t items = info.channels * info.frames;
    std::vector<double> raw(items);
    sf_read_double(file, &raw[0], items);

    for (size_t ch = 0; ch < size_t(info.channels); ++ch) {
      if (data[ch].bufSize < static_cast<size_t>(info.frames)) {
        std::cerr << "Error: FFTW3 buffer size is insufficient to load audio file.\n";
        exit(EXIT_FAILURE);
      }
      for (sf_count_t i = 0; i < info.frames; ++i) {
        data[ch].buf[i] = raw[info.channels * i + ch];
      }
    }
  }

  void load(std::vector<double> &data)
  {
    if (!file) {
      std::cerr << "Error: SoundFile::load is called before SoundFile::open.\n";
      return;
    }

    sf_count_t items = info.channels * info.frames;
    data.resize(items);
    sf_read_double(file, &data[0], items);
  }

  void close()
  {
    if (file) {
      if (sf_close(file) != 0) std::cerr << "Error: sf_close failed.\n";
    }
  }

  ~SoundFile() { close(); }
};

int writeWave(std::string path, std::vector<FFTW3Buffer> &data, const SF_INFO &inputInfo)
{
  SF_INFO info = inputInfo;
  info.format = (SF_FORMAT_WAV | SF_FORMAT_FLOAT);

  std::vector<float> raw(info.channels * info.frames);
  for (size_t idx = 0; idx < size_t(info.frames); ++idx) {
    for (size_t ch = 0; ch < size_t(info.channels); ++ch) {
      raw[info.channels * idx + ch] = static_cast<float>(data[ch].buf[idx]);
    }
  }

  SNDFILE *file = sf_open(path.c_str(), SFM_WRITE, &info);
  if (!file) {
    std::cerr << "Error: sf_open failed.\n";
    return EXIT_FAILURE;
  }

  if (sf_write_float(file, &raw[0], raw.size()) != sf_count_t(raw.size()))
    std::cerr << "Error: " << sf_strerror(file) << "\n";

  if (sf_close(file) != 0) {
    std::cerr << "Error: sf_close failed.\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int writeWave(
  std::string path,
  std::vector<double> &data,
  const sf_count_t frames,
  const sf_count_t offset,
  const SF_INFO &inputInfo)
{
  SF_INFO info = inputInfo;
  info.frames = frames;
  info.format = (SF_FORMAT_WAV | SF_FORMAT_FLOAT);

  SNDFILE *file = sf_open(path.c_str(), SFM_WRITE, &info);
  if (!file) {
    std::cerr << "Error: sf_open failed.\n";
    return EXIT_FAILURE;
  }

  // Beware that values in `info` becomes invalid after sf_open.

  auto items = inputInfo.channels * frames;
  if (sf_write_double(file, &data[offset], items) != sf_count_t(items)) {
    std::cerr << "Error: " << sf_strerror(file) << "\n";
  }

  if (sf_close(file) != 0) {
    std::cerr << "Error: sf_close failed.\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

template<typename T> struct LimiterParameter {
  T attackSeconds = T(64.0 / 48000.0);
  T sustainSeconds = T(64.0 / 48000.0);
  T releaseSeconds = 0;
  T thresholdDecibel = T(-0.1);
  T gateDecibel = -std::numeric_limits<T>::infinity();
  T link = T(0.5);
  size_t upsample = 1;
  size_t maxiter = 4;
};

struct UserOption {
  std::filesystem::path inputPath;
  std::filesystem::path outputPath;
  bool verbose = false;
  bool skipPrompt = false;
  bool isYes = false;
  bool trim = false;
  double memoryWarningThreshold = 1.0;
};

std::filesystem::path deriveOutputPath(const std::filesystem::path &input)
{
  std::string filename = "limited_" + input.filename().string();
  return input.parent_path() / std::filesystem::path(filename);
}

template<typename T> bool isValidSecond(T value)
{
  // With `/fp:fast` or `-ffast-math`, `isfinite` will likely return true in any case.
  // That's why `std::numeric_limits<T>::max()` condition is placed. Also, denormal
  // positive value is not valid.

  using nl = std::numeric_limits<T>;

  return std::isfinite(value) && (value >= nl::min() || value == T(0))
    && value <= nl::max();
}

double getPeakAmplitude(std::vector<FFTW3Buffer> &data, size_t length, bool verbose)
{
  size_t maxPeakChannel = 0;
  double maxPeak = 0.0;

  for (size_t ch = 0; ch < data.size(); ++ch) {
    auto peak = data[ch].peakAmplitude(length);

    if (verbose) {
      std::cout << std::format("Ch.{:02d} : {} [dB], {}\n", ch, ampToDecibel(peak), peak);
    }

    if (peak > maxPeak) {
      maxPeak = peak;
      maxPeakChannel = ch;
    }
  }
  if (verbose) {
    std::cout << std::format(
      "Max peak at ch.{}, {} [dB], or {} in amplitude.\n", maxPeakChannel,
      ampToDecibel(maxPeak), maxPeak);
  }
  return maxPeak;
}

double getPeakAmplitude(
  std::vector<double> &data, const size_t channels, const size_t frames, bool verbose)
{
  std::vector<double> peaks(channels);

  for (size_t idx = 0; idx < frames; ++idx) {
    for (size_t ch = 0; ch < channels; ++ch) {
      peaks[ch] = std::max(peaks[ch], std::abs(data[channels * idx + ch]));
    }
  }

  double maxPeak = 0.0;
  size_t maxPeakChannel = 0;
  for (size_t ch = 0; ch < peaks.size(); ++ch) {
    if (verbose) {
      std::cout << std::format(
        "Ch.{:02d} : {} [dB], {}\n", ch, ampToDecibel(peaks[ch]), peaks[ch]);
    }
    if (peaks[ch] > maxPeak) {
      maxPeak = peaks[ch];
      maxPeakChannel = ch;
    }
  }
  if (verbose) {
    std::cout << std::format(
      "Max peak at ch.{}, {} [dB], or {} in amplitude.\n", maxPeakChannel,
      ampToDecibel(maxPeak), maxPeak);
  }
  return maxPeak;
}

int promptMemoryUsage(double estimatedMemoryUsage, UserOption &opt)
{
  std::cout << std::format(
    "\nRequired Memory Estimation : {:.3f} [GiB]\n", estimatedMemoryUsage);

  if (estimatedMemoryUsage >= opt.memoryWarningThreshold) {
    std::cerr << std::format(
      "Warning: Required memory exceeds {} GiB.\n", opt.memoryWarningThreshold);

    if (opt.skipPrompt) {
      if (!opt.isYes) {
        std::cerr << "Info: Processing is terminated by --prompt no.\n";
        return EXIT_FAILURE;
      }
    } else {
      std::cout << "Proceed? [Y/n] " << std::flush;
      char yes = 0;
      std::cin >> yes;
      if (yes != 'y' && yes != 'Y') return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

int processMemoryEfficientMode(
  UserOption &opt, SoundFile &snd, LimiterParameter<double> &param)
{
  if (opt.verbose) {
    auto inputSize
      = sizeof(double) * snd.info.channels * snd.info.frames / double(1024 * 1024 * 1024);

    if (promptMemoryUsage(inputSize, opt) == EXIT_FAILURE) return EXIT_FAILURE;
  }

  // Load file.
  std::vector<double> data;
  snd.load(data);
  snd.close();

  const size_t channels = static_cast<size_t>(snd.info.channels);
  const size_t frames = static_cast<size_t>(snd.info.frames);

  if (opt.verbose) std::cout << "### Input Peaks\n";
  getPeakAmplitude(data, channels, frames, opt.verbose);

  // Apply limiter.
  using HeCoef = HighEliminatorCoefficient<double>;
  using UpCoef = UpSamplerCoefficient<double>;
  using DownCoef = DownSamplerCoefficient<double>;

  const auto limiterLatency
    = Limiter<double>::latency(snd.info.samplerate, param.attackSeconds);
  const auto margin = limiterLatency
    + 2 * (HeCoef::fir.size() + UpCoef::bufferSize + DownCoef::bufferSize);

  data.resize(data.size() + channels * margin);

  std::vector<OverlapSaveConvolver> highEliminators(channels);
  for (auto &he : highEliminators) {
    he.init(HeCoef::fir.size(), HeCoef::delay);
    he.setFir(&HeCoef::fir[0], 0, HeCoef::fir.size());
    he.reset();
  }

  std::vector<FirUpSampler<double, UpCoef>> upSampler(channels);
  std::vector<FirDownSampler<double, DownCoef>> downSampler(channels);

  std::vector<Limiter<double>> limiters(channels);
  const double upRate = static_cast<double>(UpCoef::upfold * snd.info.samplerate);
  for (auto &lm : limiters) {
    lm.prepare(
      upRate, param.attackSeconds, param.sustainSeconds, param.releaseSeconds,
      decibelToAmp(param.thresholdDecibel), decibelToAmp(param.gateDecibel));
  }

  if (opt.verbose) std::cout << "### Applying limiter\n";

  for (size_t frm = 0; frm < frames + margin; ++frm) {
    size_t index = channels * frm;

    for (size_t ch = 0; ch < channels; ++ch) {
      upSampler[ch].process(highEliminators[ch].process(data[index + ch]));
    }

    for (size_t jdx = 0; jdx < UpCoef::upfold; ++jdx) {
      auto maxAbs = 0.0;
      for (size_t ch = 0; ch < channels; ++ch) {
        maxAbs = std::max(maxAbs, std::abs(upSampler[ch].output[jdx]));
      }

      for (size_t ch = 0; ch < channels; ++ch) {
        auto &sig = upSampler[ch].output[jdx];
        sig = limiters[ch].process(sig, std::lerp(std::abs(sig), maxAbs, param.link));
      }
    }

    for (size_t ch = 0; ch < channels; ++ch) {
      data[index + ch] = downSampler[ch].process(upSampler[ch].output);
    }
  }

  if (opt.verbose) std::cout << "### Output Peaks\n";
  getPeakAmplitude(data, channels, frames, opt.verbose);

  // Even when `--trim` is not specified, silence introduced by FIR group delay is
  // trimmed. See the above comment on latency about -3.
  sf_count_t overlapAddLatency
    = HeCoef::fir.size() + UpCoef::bufferSize + DownCoef::bufferSize;
  sf_count_t firLatency = HeCoef::delay + UpCoef::intDelay + DownCoef::intDelay - 3;
  sf_count_t totalLatency = limiterLatency + overlapAddLatency + firLatency;
  sf_count_t offset = opt.trim ? totalLatency : firLatency;
  sf_count_t trimedFrames = opt.trim ? frames : frames + margin - firLatency;
  return writeWave(
    opt.outputPath.string(), data, trimedFrames, snd.info.channels * offset, snd.info);
}

int processPreciseMode(UserOption &opt, SoundFile &snd, LimiterParameter<double> &param)
{
  if (opt.verbose) {
    auto inputSize
      = sizeof(double) * snd.info.channels * snd.info.frames / double(1024 * 1024 * 1024);
    auto estimatedMemoryUsage = 2 * param.upsample * inputSize;

    if (promptMemoryUsage(estimatedMemoryUsage, opt) == EXIT_FAILURE) return EXIT_FAILURE;
  }

  // Up-sampling.
  const auto latency = Limiter<double>::latency(
    static_cast<double>(param.upsample * snd.info.samplerate), param.attackSeconds);
  const auto bufferSize = param.upsample * snd.info.frames + latency;
  std::vector<FFTW3Buffer> data;
  data.resize(snd.info.channels);

  for (auto &dt : data) dt.allocate(bufferSize);

  snd.load(data);
  snd.close();
  if (opt.verbose) std::cout << "### Input Peaks\n";
  getPeakAmplitude(data, static_cast<size_t>(snd.info.frames), opt.verbose);

  for (size_t iteration = 0; iteration < param.maxiter; ++iteration) {
    if (opt.verbose) {
      std::cout << std::format("## Iteration {}\n### Up-sampling\n", iteration);
    }
    for (auto &dt : data) dt.upSample(param.upsample, snd.info.frames);
    if (opt.verbose) std::cout << "### Almost True Peaks\n";
    auto upPeak = getPeakAmplitude(
      data, static_cast<size_t>(param.upsample * snd.info.frames), opt.verbose);

    auto thresholdAmp = decibelToAmp(param.thresholdDecibel);
    if (upPeak <= thresholdAmp) {
      auto outputPathStr = opt.outputPath.string();

      std::cerr << std::format(
        "Info: Peak is below threshold. Skipping {}\n", outputPathStr);

      snd.open(opt.inputPath.string());
      snd.load(data);
      return writeWave(outputPathStr, data, snd.info);
    }

    // Apply limiter.
    std::vector<Limiter<double>> limiters(data.size());
    for (auto &lm : limiters) {
      lm.prepare(
        static_cast<double>(param.upsample * snd.info.samplerate), param.attackSeconds,
        param.sustainSeconds, param.releaseSeconds, thresholdAmp,
        decibelToAmp(param.gateDecibel));
    }

    if (opt.verbose) std::cout << "### Applying limiter\n";
    for (size_t idx = 0; idx < bufferSize; ++idx) {
      // Stereo (or multi-channel) link.
      auto maxAbs = 0.0;
      for (size_t ch = 0; ch < data.size(); ++ch) {
        maxAbs = std::max(maxAbs, std::abs(data[ch].buf[idx]));
      }

      // Finally, the signal goes into limiter.
      for (size_t ch = 0; ch < data.size(); ++ch) {
        auto &sig = data[ch].buf[idx];
        sig = limiters[ch].process(sig, std::lerp(std::abs(sig), maxAbs, param.link));
      }
    }

    // Down-sampling.
    if (opt.verbose) std::cout << "### Down-sampling\n";
    for (auto &dt : data) dt.downSample(param.upsample, snd.info.frames, latency);
    if (opt.verbose) std::cout << "### Output Peaks\n";
    auto processedPeak
      = getPeakAmplitude(data, static_cast<size_t>(snd.info.frames), opt.verbose);

    // `65535 / 65536 == (2^16 - 1) / 2^16` is a number slightly below dynamic range.
    // Setting the peak exactly at 1.0 may increases the risk of true-peak clipping.
    if (processedPeak <= std::max(65535.0 / 65536.0, thresholdAmp)) break;
    if (iteration == param.maxiter - 1) {
      std::cerr << "Warning: Limiting still failed at maximum iteration count.\n";
    }
  }

  // Write to file.
  return writeWave(opt.outputPath.string(), data, snd.info);
}

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()
    .
    operator()("help,h", "Show this message.")
    .
    operator()("verbose,v", "Show processing status.")
    .
    operator()(
      "prompt,p", po::value<std::string>(),
      "Answer and skip prompt when value is set to \"yes\" or \"no\". "
      "Otherwise, prompt will show up.")
    .
    operator()("input,i", po::value<std::string>(), "Input audio file path.")
    .
    operator()("output,o", po::value<std::string>(), "Output audio file path.")
    .
    operator()(
      "upsample,u", po::value<size_t>()->default_value(1),
      "Up-sampling ratio. When set to 1, FIR polyphase up-sampling is used. When set "
      "to greater than 1, FFT up-sampling is used. FFT up-sampling requires large "
      "amount of memory that is multiple of input file size and up-sampling ratio. If "
      "FFT up-sampling is enabled and up-sampled peak is below threshold, processing "
      "will be skipped. Recommend to set to 16 or greater for precise true-peak "
      "limiting.")
    .
    operator()(
      "trim",
      "--trim has no effect when --upsample is set to greater than 1. When specified, "
      "input frame count and output frame count become the same, by trimming artifacts "
      "introduced by multirate processing. When not specified, output signal becomes "
      "longer than input signal. Additional frame count is "
      "(2560 + attack * samplerate) at front, and 1286 at back. Theoretically, trimmed "
      "signal is no longer true-peak limited.")
    .
    operator()(
      "memory,m", po::value<double>()->default_value(1.0),
      "Memory warning threshold in GiB. When estimated memory allocation exceeds "
      "this value, prompt will show up.")
    .
    operator()(
      "maxiter", po::value<size_t>()->default_value(4),
      "Maximum iteration count for additional stage limiting. Sometimes the result of "
      "true-peak limiting still exceeds the threshold. It's hard to predict the final "
      "sample-peak before donw-sampling. (If you know the method, please let me know!) "
      "Therefore offlinelimiter applies extra stage limiting in case of insufficient "
      "limiting. Loop continues until the final sample-peak becomes below 0 dB, or "
      "iteration count reaches --maxiter.")
    .
    operator()(
      "attack,a", po::value<double>()->default_value(64.0 / 48000.0),
      "Attack time in seconds.")
    .
    operator()(
      "sustain,s", po::value<double>()->default_value(64.0 / 48000.0),
      "Sustain time in seconds.")
    .
    operator()(
      "release,r", po::value<double>()->default_value(0.0), "Release time in seconds.")
    .
    operator()(
      "threshold,t", po::value<double>()->default_value(-0.1),
      "Limiter threshold in decibel.")
    .
    operator()(
      "gate,g",
      po::value<double>()->default_value(-std::numeric_limits<double>::infinity()),
      "Gate threshold in decibel.")
    .
    operator()(
      "link,l", po::value<double>()->default_value(0.5),
      "Stereo or multi-channel link amount in [0.0, 1.0]. 0.0 is no link, and "
      "1.0 is full link.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Load arguments not directly related to limiter.
  UserOption opt;
  opt.verbose = vm.count("verbose");
  if (opt.verbose) std::cout << "---\n\n";

  if (vm.count("prompt")) {
    auto answer = vm["prompt"].as<std::string>();
    if (answer == "yes") {
      opt.skipPrompt = true;
      opt.isYes = true;
    } else if (answer == "no") {
      opt.skipPrompt = true;
      opt.isYes = false;
    } else {
      std::cerr << "Warning: --prompt is not set to \"yes\" or \"no\".\n";
    }
  }

  opt.memoryWarningThreshold = vm["memory"].as<double>();
  if (std::isnan(opt.memoryWarningThreshold) || opt.memoryWarningThreshold < 0.0) {
    std::cerr << "Error: memory warning threshold must be positive.\n";
    return EXIT_FAILURE;
  }

  if (!vm.count("input") || vm.count("help")) {
    std::cout << desc << "\n";
    return EXIT_FAILURE;
  }
  opt.inputPath = std::filesystem::path(vm["input"].as<std::string>());

  opt.outputPath = vm.count("output")
    ? std::filesystem::path(vm["output"].as<std::string>())
    : deriveOutputPath(opt.inputPath);
  if (std::filesystem::exists(opt.outputPath)) {
    std::cerr << std::format("Warning: {} already exists.\n", opt.outputPath.string());

    if (opt.skipPrompt) {
      if (!opt.isYes) {
        std::cerr << "Info: Processing is terminated by --prompt no.\n";
        return EXIT_FAILURE;
      }
    } else {
      std::cout << "Overwrite? [Y/n] " << std::flush;
      char yes = 0;
      std::cin >> yes;
      if (yes != 'y' && yes != 'Y') return EXIT_SUCCESS;
    }
  }

  opt.trim = vm.count("trim");

  // Load arguments related to limiter.
  LimiterParameter<double> param;
  param.attackSeconds = vm["attack"].as<double>();
  param.sustainSeconds = vm["sustain"].as<double>();
  param.releaseSeconds = vm["release"].as<double>();
  param.thresholdDecibel = vm["threshold"].as<double>();
  param.gateDecibel = vm["gate"].as<double>();
  param.link = vm["link"].as<double>();
  param.upsample = vm["upsample"].as<size_t>();
  param.maxiter = vm["maxiter"].as<size_t>();

  bool isInvalid = false;
  if (!isValidSecond(param.attackSeconds)) {
    std::cerr << "Error: Attack time must be positive value in seconds.\n";
    isInvalid = true;
  }
  if (!isValidSecond(param.sustainSeconds)) {
    std::cerr << "Error: Sustain time must be positive value in seconds.\n";
    isInvalid = true;
  }
  if (!isValidSecond(param.releaseSeconds)) {
    std::cerr << "Error: Release time must be positive value in seconds.\n";
    isInvalid = true;
  }
  if (std::isnan(param.thresholdDecibel)) {
    std::cerr << "Error: Limiter threshold must not be NaN.\n";
    isInvalid = true;
  }
  if (std::isnan(param.gateDecibel)) {
    std::cerr << "Error: Gate threshold must not be NaN.\n";
    isInvalid = true;
  }
  if (std::isnan(param.link) || param.link < 0.0 || param.link > 1.0) {
    std::cerr << "Error: Link amount must be in [0.0, 1.0].\n";
    isInvalid = true;
  }
  if (param.upsample <= 0) {
    std::cerr << "Error: Up-sampling ratio must be greater than 0.\n";
    isInvalid = true;
  }
  if (param.upsample > 1 && opt.trim) {
    std::cerr << "Warning: --trim has no effect when --upsample is greater than 1.\n";
  }
  if (param.maxiter <= 0) {
    std::cerr << "Error: Maximum iteration count must be greater than 0.\n";
    isInvalid = true;
  }
  if (isInvalid || opt.verbose) {
    std::cout << std::format(
      R"(Input path  : {}
Output path : {}
Mode        : {}
Trim        : {}
Attack      : {} [s]
Sustain     : {} [s]
Release     : {} [s]
Threshold   : {} [dB]
Gate        : {} [dB]
Link        : {}
Up-sampling : {}
)",
      opt.inputPath.string(), opt.outputPath.string(),
      param.upsample > 1 ? "FFT (precise)" : "FIR Polyphase (non-precise)", opt.trim,
      param.attackSeconds, param.sustainSeconds, param.releaseSeconds,
      param.thresholdDecibel, param.gateDecibel, param.link, param.upsample);
  }
  if (isInvalid) return EXIT_FAILURE;

  // Load audio file infomation.
  SoundFile snd;
  if (snd.open(opt.inputPath.string()) == EXIT_FAILURE) return EXIT_FAILURE;

  if (opt.verbose) {
    std::cout << std::format(
      R"(Sample Rate : {} [Hz]
Channel     : {}
Frame       : {}
)",
      snd.info.samplerate, snd.info.channels, snd.info.frames);
  }

  // Processing.
  auto start = std::chrono::steady_clock::now();
  auto exitCode = param.upsample > 1 ? processPreciseMode(opt, snd, param)
                                     : processMemoryEfficientMode(opt, snd, param);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  if (opt.verbose) {
    std::cout << std::format("Elapsed Time : {} [s]\n", elapsed.count());
  }
  return exitCode;
}
