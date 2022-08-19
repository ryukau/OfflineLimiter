#
# Note: Using libraries through vcpkg.
#

if (!(test-path build)) {
  cmake -S . -B build `
    -DCMAKE_TOOLCHAIN_FILE="/src/vcpkg/scripts/buildsystems/vcpkg.cmake" `
    -DCMAKE_BUILD_TYPE=Release
}

cmake --build build --config debug
./build/Debug/offlinelimiter.exe
# python plot.py
