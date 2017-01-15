# general dependencies
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev

# protobuf (compile from source)
wget https://github.com/google/protobuf/releases/download/v3.0.0/protobuf-cpp-3.0.0.tar.gz -O ../protobuf-cpp-3.0.0.tar.gz
pushd ..
tar xvf protobuf-cpp-3.0.0.tar.gz
cd protobuf-3.0.0
./configure --disable-shared
read -p "add -fPIC in CXXFLAGS, protobuf-3.0.0/Makefile ..." -n1 -s
printf "\ncompiling protobuf\n"
make
make check
sudo make install
sudo ldconfig
popd

# OpenBLAS (compile from source)
git clone https://github.com/xianyi/OpenBLAS ~/OpenBLAS
pushd ~/OpenBLAS
printf "\ncompiling OpenBLAS\n"
make
sudo make install

# CUDA & CuDNN
echo "choose CUDA version (if you dont have latest drivers, like in gpu-srv, don't choose cuda-8.0) :"
echo "1 - CUDA 7.5"
echo "2 - CUDA 8.0"
read cuda_opt

printf "\ninstalling CUDA & CuDNN\n"
if [ "$cuda_opt" = "1" ] then
    # CUDA 7.5 & CUDNN 5.1
    sudo sh /server/software/NVIDIA/cuda/cuda_7.5.18_linux.run
    cd /usr/local
    sudo tar xvf /server/software/NVIDIA/cudnn/cudnn-7.5-linux-x64-v5.1.tgz
else
    # CUDA 8.0 & CUDNN 5.1
    sudo sh /server/software/NVIDIA/cuda/cuda_8.0.44_linux.run
    cd /usr/local
    sudo tar xvf /server/software/NVIDIA/cudnn/cudnn-8.0-linux-x64-v5.1.tgz
fi

# Caffe
cp setup/Makefile.config ../caffe_ristretto_Austriker
cd ../caffe_ristretto_Austriker
#read -p "go into caffe Makefile and : (1) remove protobuf from LIBRARIES. (2) add line 'LDFLAGS += -Wl,-Bstatic -lprotobuf -Wl,-Bdynamic'" -n1 -s
printf "\ncompiling caffe\n"
make all
make matcaffe
make pycaffe