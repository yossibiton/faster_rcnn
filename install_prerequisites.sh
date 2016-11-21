# OpenBLAS
git clone https://github.com/xianyi/OpenBLAS ~/OpenBLAS
pushd ~/OpenBLAS
make
sudo make install
popd

echo "choose CUDA version (if you dont have latest drivers, like in gpu-srv, don't choose cuda-8.0) :"
echo "1 - CUDA 7.5"
echo "2 - CUDA 8.0"
read cuda_opt

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
cd ../caffe_Austriker
make all
make matcaffe