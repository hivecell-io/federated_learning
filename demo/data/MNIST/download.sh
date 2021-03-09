apt-get update && apt-get install -y wget p7zip-full

mkdir -p /workspace/demo/data/MNIST/raw
mkdir -p /workspace/demo/data/MNIST/processed

cd /workspace/demo/data/MNIST/raw

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

7z x train-images-idx3-ubyte.gz
7z x train-labels-idx1-ubyte.gz
7z x t10k-images-idx3-ubyte.gz
7z x t10k-labels-idx1-ubyte.gz

cd ..

python3 /workspace/demo/data/MNIST/process.py
