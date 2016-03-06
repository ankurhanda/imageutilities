#ImageUtilities 

mkdir build

cd build 

cmake .. -DCUDA_PROPAGATE_HOST_FLAGS=0

make -j8 (it will throw lots of warnings but will be fixed soon!)
