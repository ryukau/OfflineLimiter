#include "limiter.hpp"

#include <cstdio>
#include <cstdlib>
#include <format>

void testLimiter()
{
  double fs = 48000.0;
  size_t attackSample = 64;
  size_t sustainSample = 64;
  double releaseSecond = 0.001;
  double thresholdAmplitude = 1.0;
  double gateAmplitude = 0.0;

  Limiter<double> limiter;

  limiter.resize(attackSample, sustainSample);
  limiter.prepare(
    fs, attackSample / fs, sustainSample / fs, releaseSecond, thresholdAmplitude,
    gateAmplitude);

  auto value = limiter.process(1, 1);

  std::printf("%f\n", value);
}

int main()
{
  testLimiter();

  return EXIT_SUCCESS;
}
