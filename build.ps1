#
# Note: Using libraries through vcpkg.
#

if (!(test-path build)) {
  cmake -S . -B build `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_TOOLCHAIN_FILE="/src/vcpkg/scripts/buildsystems/vcpkg.cmake" `
    -DCMAKE_BUILD_TYPE=Release
}

cmake --build build --config release
./build/Release/offlinelimiter.exe
