cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(offlinelimiter)
add_executable(${PROJECT_NAME}
  offlinelimiter.cpp
  polyphase.hpp
  limiter.hpp
  fir/downsamplercoefficient.hpp
  fir/higheliminatorcoefficient.hpp
  fir/upsamplercoefficient.hpp
)
target_compile_options(${PROJECT_NAME} PRIVATE
  /W4
  /fp:fast
  # /Qvec-report:1
)

# Packages.
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(SndFile CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)

# Link.
target_link_libraries(${PROJECT_NAME} PRIVATE
  Boost::program_options
  FFTW3::fftw3
  SndFile::sndfile
)
