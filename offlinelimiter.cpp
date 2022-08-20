/*

TODO:
- Add time spent.
- Add second stage.
- Maybe add better progress text.
- Maybe add output file format option.

*/

#include "fir/downsamplercoefficient.hpp"
#include "fir/higheliminatorcoefficient.hpp"
#include "fir/upsamplercoefficient.hpp"
#include "limiter.hpp"
#include "polyphase.hpp"

#include <boost/program_options.hpp>
#include <fftw3.h>
#include <sndfile.h>

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
      std::cerr << "Error: sf_open failed." << std::endl;
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

  void load(std::vector<FFTW3Buffer> &data)
  {
    if (!file) {
      std::cerr << "Error: SoundFile::load is called before SoundFile::open."
                << std::endl;
      return;
    }

    sf_count_t items = info.channels * info.frames;
    std::vector<double> raw(items);
    sf_read_double(file, &raw[0], items);

    for (size_t ch = 0; ch < size_t(info.channels); ++ch) {
      if (data[ch].bufSize < static_cast<size_t>(info.frames)) {
        std::cerr << "Error: FFTW3 buffer size is insufficient to load audio file.";
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
      std::cerr << "Error: SoundFile::load is called before SoundFile::open."
                << std::endl;
      return;
    }

    sf_count_t items = info.channels * info.frames;
    data.resize(items);
    sf_read_double(file, &data[0], items);
  }

  void close()
  {
    if (file) {
      if (sf_close(file) != 0) std::cerr << "Error: sf_close failed." << std::endl;
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
    std::cerr << "Error: sf_open failed." << std::endl;
    return EXIT_FAILURE;
  }

  if (sf_write_float(file, &raw[0], raw.size()) != sf_count_t(raw.size()))
    std::cerr << "Error: " << sf_strerror(file) << std::endl;

  if (sf_close(file) != 0) {
    std::cerr << "Error: sf_close failed." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int writeWave(
  std::string path,
  std::vector<double> &data,
  const size_t latency,
  const SF_INFO &inputInfo)
{
  SF_INFO info = inputInfo;
  info.format = (SF_FORMAT_WAV | SF_FORMAT_FLOAT);

  auto items = info.channels * info.frames;
  size_t offset = static_cast<size_t>(info.channels) * latency;

  SNDFILE *file = sf_open(path.c_str(), SFM_WRITE, &info);
  if (!file) {
    std::cerr << "Error: sf_open failed." << std::endl;
    return EXIT_FAILURE;
  }

  if (sf_write_double(file, &data[offset], items) != sf_count_t(items)) {
    std::cerr << "Error: " << sf_strerror(file) << std::endl;
  }

  if (sf_close(file) != 0) {
    std::cerr << "Error: sf_close failed." << std::endl;
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
  size_t fold = 16;
};

struct UserOption {
  std::filesystem::path outputPath;
  bool isVerbose = false;
  bool skipPrompt = false;
  bool isYes = false;
  double memoryWarningThreshold = 1.0;
};

std::filesystem::path deriveOutputPath(const std::string &input)
{
  std::filesystem::path path(input);
  std::string filename = "limited_" + path.filename().string();
  return path.parent_path() / std::filesystem::path(filename);
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

void printPeakAmplitude(std::vector<FFTW3Buffer> &data, size_t length)
{
  size_t maxPeakChannel = 0;
  double maxPeak = 0.0;

  for (size_t ch = 0; ch < data.size(); ++ch) {
    auto peak = data[ch].peakAmplitude(length);

    std::cout << std::format("Ch.{:02d} : {} [dB], {}\n", ch, ampToDecibel(peak), peak);

    if (peak > maxPeak) {
      maxPeak = peak;
      maxPeakChannel = ch;
    }
  }
  std::cout << std::format(
    "Max peak at ch.{}, {} [dB], or {} in amplitude.\n\n", maxPeakChannel,
    ampToDecibel(maxPeak), maxPeak);
}

void printPeakAmplitude(
  std::vector<double> &data, const size_t channels, const size_t frames)
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
    std::cout << std::format(
      "Ch.{:02d} : {} [dB], {}\n", ch, ampToDecibel(peaks[ch]), peaks[ch]);
    if (peaks[ch] > maxPeak) {
      maxPeak = peaks[ch];
      maxPeakChannel = ch;
    }
  }
  std::cout << std::format(
    "Max peak at ch.{}, {} [dB], or {} in amplitude.\n\n", maxPeakChannel,
    ampToDecibel(maxPeak), maxPeak);
}

int promptMemoryUsage(double estimatedMemoryUsage, UserOption &opt)
{
  std::cout << std::format(
    "\nRequired Memory Estimation : {:.3f} [GB]\n", estimatedMemoryUsage);

  if (estimatedMemoryUsage >= opt.memoryWarningThreshold) {
    std::cerr << std::format(
      "Warning: Required memory exceeds {} GiB.\n", opt.memoryWarningThreshold);

    if (opt.skipPrompt) {
      if (!opt.isYes) {
        if (opt.isVerbose) {
          std::cout << "Info: Processing is terminated by --prompt no.\n";
        }
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
  if (opt.isVerbose) {
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

  if (opt.isVerbose) {
    std::cout << "\nInput Peaks\n";
    printPeakAmplitude(data, channels, frames);
  }

  // Apply limiter.
  using HeCoef = HighEliminatorCoefficient<double>;
  using UpCoef = UpSamplerCoefficient<double>;
  using DownCoef = DownSamplerCoefficient<double>;

  const double upRate = static_cast<double>(UpCoef::upfold * snd.info.samplerate);
  const auto latency = Limiter<double>::latency(upRate, param.attackSeconds)
    + HeCoef::delay + UpCoef::intDelay + DownCoef::intDelay;
  data.resize(data.size() + channels * latency);

  std::vector<NaiveConvolver<double, HeCoef>> highEliminators(channels);
  std::vector<FirUpSampler<double, UpCoef>> upSampler(channels);
  std::vector<FirDownSampler<double, DownCoef>> downSampler(channels);

  std::vector<Limiter<double>> limiters(channels);
  for (auto &lm : limiters) {
    lm.prepare(
      upRate, param.attackSeconds, param.sustainSeconds, param.releaseSeconds,
      decibelToAmp(param.thresholdDecibel), decibelToAmp(param.gateDecibel));
  }

  if (opt.isVerbose) std::cout << "Applying limiter\n";

  for (size_t frm = 0; frm < frames; ++frm) {
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

  if (opt.isVerbose) {
    std::cout << "\nOutput Peaks\n";
    printPeakAmplitude(data, channels, frames);
  }

  return writeWave(opt.outputPath.string(), data, latency, snd.info);
}

int processPreciseMode(UserOption &opt, SoundFile &snd, LimiterParameter<double> &param)
{
  if (opt.isVerbose) {
    auto inputSize
      = sizeof(double) * snd.info.channels * snd.info.frames / double(1024 * 1024 * 1024);
    auto estimatedMemoryUsage = 2 * param.fold * inputSize;

    if (promptMemoryUsage(estimatedMemoryUsage, opt) == EXIT_FAILURE) return EXIT_FAILURE;
  }

  // Up-sampling.
  const auto latency = Limiter<double>::latency(
    static_cast<double>(param.fold * snd.info.samplerate), param.attackSeconds);
  const auto bufferSize = param.fold * snd.info.frames + latency;
  std::vector<FFTW3Buffer> data;
  data.resize(snd.info.channels);

  for (auto &dt : data) dt.allocate(bufferSize);

  snd.load(data);
  snd.close();
  if (opt.isVerbose) {
    std::cout << "\nInput Peaks\n";
    printPeakAmplitude(data, static_cast<size_t>(snd.info.frames));
  }

  if (opt.isVerbose) std::cout << "Up-sampling\n";
  for (auto &dt : data) dt.upSample(param.fold, snd.info.frames);
  if (opt.isVerbose) {
    std::cout << "\nAlmost True Peaks\n";
    printPeakAmplitude(data, static_cast<size_t>(param.fold * snd.info.frames));
  }

  // Apply limiter.
  std::vector<Limiter<double>> limiters(data.size());
  for (auto &lm : limiters) {
    lm.prepare(
      static_cast<double>(param.fold * snd.info.samplerate), param.attackSeconds,
      param.sustainSeconds, param.releaseSeconds, decibelToAmp(param.thresholdDecibel),
      decibelToAmp(param.gateDecibel));
  }

  if (opt.isVerbose) std::cout << "Applying limiter\n";

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
  if (opt.isVerbose) std::cout << "Down-sampling\n";
  for (auto &dt : data) dt.downSample(param.fold, snd.info.frames, latency);
  if (opt.isVerbose) {
    std::cout << "\nOutput Peaks\n";
    printPeakAmplitude(data, static_cast<size_t>(snd.info.frames));
  }

  // Write to file.
  return writeWave(opt.outputPath.string(), data, snd.info);
}

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()                                                                   //
    ("help,h",                                                                         //
     "Show this message.")                                                             //
    ("verbose,v",                                                                      //
     "Show processing status.")                                                        //
    ("prompt,p",                                                                       //
     po::value<std::string>(),                                                         //
     "Answer and skip prompt when value is set to \"yes\" or \"no\". "                 //
     "Otherwise, prompt will show up.")                                                //
    ("memory,m",                                                                       //
     po::value<double>()->default_value(1.0),                                          //
     "Memory warning threshold in GiB. When estimated memory allocation exceeds "      //
     "this value, prompt will show up.")                                               //
    ("input,i",                                                                        //
     po::value<std::string>(),                                                         //
     "Input audio file path.")                                                         //
    ("output,o",                                                                       //
     po::value<std::string>(),                                                         //
     "Output audio file path.")                                                        //
    ("precise",                                                                        //
     "When specified, FFT up-sampling is used instead of FIR polyphase up-sampling. "  //
     "FFT up-sampling requires large amount of memory that is multiple of input file " //
     "size and up-sampling ratio.")                                                    //
    ("attack,a",                                                                       //
     po::value<double>()->default_value(64.0 / 48000.0),                               //
     "Attack time in seconds.")                                                        //
    ("sustain,s",                                                                      //
     po::value<double>()->default_value(64.0 / 48000.0),                               //
     "Sustain time in seconds.")                                                       //
    ("release,r",                                                                      //
     po::value<double>()->default_value(0.0),                                          //
     "Release time in seconds.")                                                       //
    ("threshold,t",                                                                    //
     po::value<double>()->default_value(-0.1),                                         //
     "Limiter threshold in decibel.")                                                  //
    ("gate,g",                                                                         //
     po::value<double>()->default_value(-std::numeric_limits<double>::infinity()),     //
     "Gate threshold in decibel.")                                                     //
    ("link,l",                                                                         //
     po::value<double>()->default_value(0.5),                                          //
     "Stereo or multi-channel link amount in [0.0, 1.0]. 0.0 is no link, and "         //
     "1.0 is full link.")                                                              //
    ("upsample,u",                                                                     //
     po::value<size_t>()->default_value(16),                                           //
     "Up-sampling ratio.")                                                             //
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Load arguments not directly related to limiter.
  UserOption opt;
  opt.isVerbose = vm.count("verbose");

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
  std::string inputPath{vm["input"].as<std::string>()};

  opt.outputPath = vm.count("output")
    ? std::filesystem::path(vm["output"].as<std::string>())
    : deriveOutputPath(inputPath);
  if (std::filesystem::exists(opt.outputPath)) {
    std::cerr << std::format("Warning: {} already exists.\n", opt.outputPath.string());

    if (opt.skipPrompt) {
      if (!opt.isYes) {
        if (opt.isVerbose)
          std::cout << "Info: Processing is terminated by --prompt no.\n";
        return EXIT_FAILURE;
      }
    } else {
      std::cout << "Overwrite? [Y/n] " << std::flush;
      char yes = 0;
      std::cin >> yes;
      if (yes != 'y' && yes != 'Y') return EXIT_SUCCESS;
    }
  }

  // Load arguments related to limiter.
  LimiterParameter<double> param;
  param.attackSeconds = vm["attack"].as<double>();
  param.sustainSeconds = vm["sustain"].as<double>();
  param.releaseSeconds = vm["release"].as<double>();
  param.thresholdDecibel = vm["threshold"].as<double>();
  param.gateDecibel = vm["gate"].as<double>();
  param.link = vm["link"].as<double>();
  param.fold = vm["upsample"].as<size_t>();

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
  if (param.fold <= 0) {
    std::cerr << "Error: Up-sampling ratio must be greater than 0.\n";
    isInvalid = true;
  }
  if (isInvalid || opt.isVerbose) {
    std::cout << std::format(
      R"(
Input path  : {}
Output path : {}

Attack      : {} [s]
Sustain     : {} [s]
Release     : {} [s]
Threshold   : {} [dB]
Gate        : {} [dB]
Link        : {}
Up-sampling : {}
)",
      inputPath, opt.outputPath.string(), param.attackSeconds, param.sustainSeconds,
      param.releaseSeconds, param.thresholdDecibel, param.gateDecibel, param.link,
      param.fold);
  }
  if (isInvalid) return EXIT_FAILURE;

  // Load audio file infomation.
  SoundFile snd;
  if (snd.open(inputPath) == EXIT_FAILURE) return EXIT_FAILURE;

  if (opt.isVerbose) {
    std::cout << std::format(
      R"(
Sample Rate : {} [Hz]
Channel     : {}
Frame       : {}
)",
      snd.info.samplerate, snd.info.channels, snd.info.frames);
  }

  if (vm.count("precise")) {
    return processPreciseMode(opt, snd, param);
  }
  return processMemoryEfficientMode(opt, snd, param);
}
