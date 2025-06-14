# check_gpu.py
import torch

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")
else:
    print("경고: PyTorch가 CUDA를 찾을 수 없습니다. CPU 버전을 사용 중일 수 있습니다.")