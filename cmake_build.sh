#!/bin/sh

cdir=$(pwd)

# nvxs
cd "${cdir}/nvxs/CLAPACK"
make
cd "${cdir}/nvxs"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install

# ruby ext
cd "${cdir}/animeface-ruby"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install

echo -n "\n\nCheck:"
echo " % cd animeface-ruby"
echo " % ruby sample.rb <input image>"
