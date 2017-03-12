protobuf_repo=https://github.com/google/protobuf/releases/download/v3.0.0/protobuf-cpp-3.0.0.tar.gz
openblas_repo=https://github.com/xianyi/OpenBLAS
caffe_repo=/opt/git/caffe_ristretto_Austriker
matcaffetools_repo=https://github.com/yossibiton/matcaffe_tools
squeezenet_repo=https://github.com/yossibiton/SqueezeNet

cuda75_url=/server/software/NVIDIA/cuda/cuda_7.5.18_linux.run
cuda80_url=/server/software/NVIDIA/cudnn/cudnn-7.5-linux-x64-v5.1.tgz
cudnn75_url=/server/software/NVIDIA/cuda/cuda_8.0.44_linux.run
cudnn80_url=/server/software/NVIDIA/cudnn/cudnn-8.0-linux-x64-v5.1.tgz

## general dependencies ##
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev

## protobuf (compile from source) ##
pushd ..
wget $protobuf_repo -O protobuf-cpp-3.0.0.tar.gz
tar xvf protobuf-cpp-3.0.0.tar.gz
cd protobuf-3.0.0
./configure --disable-shared
read -p "add -fPIC in CXXFLAGS, protobuf-3.0.0/Makefile ..." -n1 -s
printf "\ncompiling protobuf\n"
make -j8
make -j8 check
sudo make install
sudo ldconfig
popd

## OpenBLAS (compile from source) ##
pushd ..
git clone $openblas_repo OpenBLAS
cd OpenBLAS
printf "\ncompiling OpenBLAS\n"
make
sudo make install
popd

## CUDA & CuDNN ##
echo "choose CUDA version (if you dont have latest drivers, like in gpu-srv, don't choose cuda-8.0) :"
echo "1 - CUDA 7.5"
echo "2 - CUDA 8.0"
echo "3 - don't install CUDA (alrteady installed)"
read cuda_opt

printf "\ninstalling CUDA & CuDNN\n"
if [ "$cuda_opt" = "1" ] then
    # CUDA 7.5 & CUDNN 5.1
    sudo sh $cuda75_url 
    cd /usr/local
    sudo tar xvf $cudnn75_url
else if [ "$cuda_opt" = "2" ] then
    # CUDA 8.0 & CUDNN 5.1
    sudo sh $cuda80_url 
    cd /usr/local
    sudo tar xvf $cudnn80_url
fi

## Caffe ##
pushd ..
git clone $caffe_repo
cd caffe_ristretto_Austriker
# TODO: The reason for doing that (using protobuf shared library) was to avoid some unexpected protobuf errors, but currently i can't compile caffe that way :
#read -p "go into caffe Makefile and : (1) remove protobuf from LIBRARIES. (2) add line 'LDFLAGS += -Wl,-Bstatic -lprotobuf -Wl,-Bdynamic'" -n1 -s
printf "\ncompiling caffe\n"
make all
make matcaffe
make pycaffe
popd

## Caffe helper libraries ##
git clone $matcaffetools_repo ../matcaffe_tools
git clone $squeezenet_repo ../SqueezeNet