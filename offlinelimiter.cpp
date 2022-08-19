/*

TODO:
- Add fast limiter path.
- Add peak indicator.
- Add time spent.
- Add second stage.
- Maybe add better progress text.

*/

#include "limiter.hpp"

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

struct FFTW3Buffer {
  double *buf = nullptr;
  std::complex<double> *spc = nullptr;

  size_t bufSize = 0;
  size_t spcSize = 0;

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
    std::cout << "Error: sf_open failed." << std::endl;
    return EXIT_FAILURE;
  }

  if (sf_write_float(file, &raw[0], raw.size()) != sf_count_t(raw.size()))
    std::cout << sf_strerror(file) << std::endl;

  if (sf_close(file) != 0) {
    std::cout << "Error: sf_close failed." << std::endl;
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

void testLimiter()
{
  double samplerate = 48000.0;
  double attackSecond = 64.0 / 48000.0;
  double sustainSample = 64.0 / 48000.0;
  double releaseSecond = 0.001;
  double thresholdAmplitude = 1.0;
  double gateAmplitude = 0.0;

  Limiter<double> limiter;

  limiter.prepare(
    samplerate, attackSecond, sustainSample, releaseSecond, thresholdAmplitude,
    gateAmplitude);

  std::cout << limiter.process(1, 1) << std::endl;
}

std::filesystem::path deriveOutputPath(const std::string &input)
{
  std::filesystem::path path(input);
  std::string filename = "limited_" + path.filename().string();
  return path.parent_path() / std::filesystem::path(filename);
}

template<typename T> bool isValidSecond(T value)
{
  // With `/fp:fast` or `-mfast-math`, `isfinite` will likely return true in any case.
  // That's why `std::numeric_limits<T>::max()` condition is placed. Also, denormal
  // positive value is not valid.

  using nl = std::numeric_limits<T>;

  return std::isfinite(value) && (value >= nl::min() || value == T(0))
    && value <= nl::max();
}

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()                                                               //
    ("help,h",                                                                     //
     "Show this message.")                                                         //
    ("verbose,v",                                                                  //
     "Show processing status.")                                                    //
    ("prompt,p",                                                                   //
     po::value<std::string>(),                                                     //
     "Answer and skip prompt when value is set. Value must be \"yes\" or \"no\". " //
     "Otherwise, prompt will show up.")                                            //
    ("memory,m",                                                                   //
     po::value<double>()->default_value(1.0),                                      //
     "Memory warning threshold in GiB. When estimated memory allocation exceeds "  //
     "this value, prompt will show up.")                                           //
    ("input,i",                                                                    //
     po::value<std::string>(),                                                     //
     "Input audio file path.")                                                     //
    ("output,o",                                                                   //
     po::value<std::string>(),                                                     //
     "Output audio file path.")                                                    //
    ("attack,a",                                                                   //
     po::value<double>()->default_value(64.0 / 48000.0),                           //
     "Attack time in seconds.")                                                    //
    ("sustain,s",                                                                  //
     po::value<double>()->default_value(64.0 / 48000.0),                           //
     "Sustain time in seconds.")                                                   //
    ("release,r",                                                                  //
     po::value<double>()->default_value(0.0),                                      //
     "Release time in seconds.")                                                   //
    ("threshold,t",                                                                //
     po::value<double>()->default_value(-0.1),                                     //
     "Limiter threshold in decibel.")                                              //
    ("gate,g",                                                                     //
     po::value<double>()->default_value(-std::numeric_limits<double>::infinity()), //
     "Gate threshold in decibel.")                                                 //
    ("link,l",                                                                     //
     po::value<double>()->default_value(0.5),                                      //
     "Stereo or multi-channel link amount in [0.0, 1.0]. 0.0 is no link, and "     //
     "1.0 is full link.")                                                          //
    ("upsample,u",                                                                 //
     po::value<size_t>()->default_value(16),                                       //
     "Up-sampling ratio.")                                                         //
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Load arguments not directly related to limiter.
  bool isVerbose = vm.count("verbose");

  bool skipPrompt = false;
  bool isYes = 0;
  if (vm.count("prompt")) {
    auto answer = vm["prompt"].as<std::string>();
    if (answer == "yes") {
      skipPrompt = true;
      isYes = true;
    } else if (answer == "no") {
      skipPrompt = true;
      isYes = false;
    } else {
      std::cerr << "Warning: --prompt is not set to \"yes\" or \"no\".\n";
    }
  }

  double memoryWarningThreshold = vm["memory"].as<double>();
  if (std::isnan(memoryWarningThreshold) || memoryWarningThreshold < 0.0) {
    std::cerr << "Error: memory warning threshold must be positive.\n";
    return EXIT_FAILURE;
  }

  if (!vm.count("input") || vm.count("help")) {
    std::cout << desc << "\n";
    return EXIT_FAILURE;
  }
  std::string inputPath{vm["input"].as<std::string>()};

  std::filesystem::path outputPath{
    vm.count("output") ? std::filesystem::path(vm["output"].as<std::string>())
                       : deriveOutputPath(inputPath)};
  if (std::filesystem::exists(outputPath)) {
    std::cerr << std::format("Warning: {} already exists.\n", outputPath.string());

    if (skipPrompt) {
      if (!isYes) {
        if (isVerbose) std::cout << "Info: Processing is terminated by --prompt no.\n";
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
  if (isInvalid || isVerbose) {
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
      inputPath, outputPath.string(), param.attackSeconds, param.sustainSeconds,
      param.releaseSeconds, param.thresholdDecibel, param.gateDecibel, param.link,
      param.fold);
  }
  if (isInvalid) return EXIT_FAILURE;

  // Load audio file infomation.
  SoundFile snd;
  if (snd.open(inputPath) == EXIT_FAILURE) return EXIT_FAILURE;

  if (isVerbose) {
    auto inputSize
      = sizeof(double) * snd.info.channels * snd.info.frames / double(1024 * 1024 * 1024);
    auto estimatedMemoryUsage = 2 * param.fold * inputSize;

    std::cout << std::format(
      R"(
Sample Rate : {} [Hz]
Channel     : {}
Frame       : {}

Required Memory Estimation : {:.3f} [GB]
)",
      snd.info.samplerate, snd.info.channels, snd.info.frames, estimatedMemoryUsage);

    if (estimatedMemoryUsage >= memoryWarningThreshold) {
      std::cerr << std::format(
        "Warning: Required memory exceeds {} GiB.\n", memoryWarningThreshold);

      if (skipPrompt) {
        if (!isYes) {
          if (isVerbose) std::cout << "Info: Processing is terminated by --prompt no.\n";
          return EXIT_FAILURE;
        }
      } else {
        std::cout << "Proceed? [Y/n] " << std::flush;
        char yes = 0;
        std::cin >> yes;
        if (yes != 'y' && yes != 'Y') return EXIT_SUCCESS;
      }
    }
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

  if (isVerbose) std::cout << "Up-sampling\n";
  for (auto &dt : data) dt.upSample(param.fold, snd.info.frames);

  // Apply limiter.
  std::vector<Limiter<double>> limiters(data.size());
  for (auto &lm : limiters) {
    lm.prepare(
      static_cast<double>(param.fold * snd.info.samplerate), param.attackSeconds,
      param.sustainSeconds, param.releaseSeconds, decibelToAmp(param.thresholdDecibel),
      decibelToAmp(param.gateDecibel));
  }

  if (isVerbose) std::cout << "Applying limiter\n";

  for (size_t idx = 0; idx < bufferSize; ++idx) {
    // Stereo (or multi-channel) link.
    auto maxAbs = 0.0;
    for (size_t ch = 0; ch < data.size(); ++ch) {
      auto absed = std::abs(data[ch].buf[idx]);
      if (maxAbs < absed) maxAbs = absed;
    }

    // Finally, the signal goes into limiter.
    for (size_t ch = 0; ch < data.size(); ++ch) {
      auto &sample = data[ch].buf[idx];
      auto linked = std::lerp(std::abs(sample), maxAbs, param.link);
      sample = limiters[ch].process(sample, linked);
    }
  }

  // Down-sampling.
  if (isVerbose) std::cout << "Down-sampling\n";
  for (auto &dt : data) dt.downSample(param.fold, snd.info.frames, latency);

  // Write to file.
  return writeWave(outputPath.string(), data, snd.info);
}
