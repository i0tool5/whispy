# Install

Use latest `faster-whisper`
```
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```
If using NVidia adapter, CUDA providers must be installed
```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Running in python virtual environment may cause `RuntimeError: Library libcublas.so.12 is not found or cannot be loaded` error, even when CUDA providers are installed. Command below fix this behavior:
```
LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.13/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH python ./src/main.py
```
Note: **python3.13 must be changed to proper python version**
