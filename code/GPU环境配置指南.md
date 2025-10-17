# GPU加速环境配置指南 (Windows & Linux)

## 📋 概述

GPU加速主要用于加速以下计算：
1. **批量光线生成** - 使用PyTorch张量运算
2. **批量功率计算** - 使用GPU并行计算

**性能提升**: CPU模式 → GPU模式 约 **8-10倍** 加速

**支持平台**:
- ✅ Windows 10/11
- ✅ Linux (Ubuntu 18.04+, CentOS 7+)

---

## 🖥️ 硬件要求

### 必需硬件
- **NVIDIA GPU**: 支持CUDA的显卡
  - 推荐: RTX 3060 及以上 (6GB+ 显存)
  - 最低: GTX 1060 (6GB 显存)
  - 不支持: AMD显卡 (当前脚本仅支持CUDA)

### 显存需求估算

| 计算规模 | 建筑三角形数 | 轨迹点数 | 推荐显存 |
|---------|------------|---------|---------|
| 小型 | < 100万 | < 1000 | 4GB |
| 中型 | 100-500万 | 1000-5000 | 6-8GB |
| 大型 | 500-1000万 | 5000-10000 | 8-12GB |
| 超大型 | > 1000万 | > 10000 | 16GB+ |

---

## 💻 环境配置 (分平台指南)

### 🪟 Windows 环境配置

#### 步骤1: 检查NVIDIA驱动

```bash
# 打开命令提示符(CMD)或PowerShell
nvidia-smi
```

**期望输出:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|...
```

**如果报错 "nvidia-smi不是内部或外部命令":**
1. 访问NVIDIA驱动下载页: https://www.nvidia.com/Download/index.aspx
2. 选择对应的显卡型号和Windows版本
3. 下载并安装驱动 (建议选择 "Game Ready Driver")
4. **重启电脑**
5. 重新运行 `nvidia-smi` 验证

#### 步骤2: 安装CUDA Toolkit (可选)

**注意**: PyTorch自带CUDA运行库，**通常不需要单独安装CUDA Toolkit**

**如果需要手动安装 (高级用户):**
1. 访问: https://developer.nvidia.com/cuda-downloads
2. 选择: `Windows` → `x86_64` → `10/11` → `exe (local)`
3. 推荐版本: **CUDA 11.8** (兼容性最好)
4. 下载并安装 (约3GB)
5. 验证安装:
   ```bash
   nvcc --version
   ```

#### 步骤3: 创建Python环境 (推荐使用Conda)

```bash
# 打开Anaconda Prompt

# 创建新环境
conda create -n pv_gpu python=3.9
conda activate pv_gpu

# 或使用venv
python -m venv pv_gpu_env
pv_gpu_env\Scripts\activate
```

#### 步骤4: 安装PyTorch (CUDA版本)

**方法A: 推荐安装 (CUDA 11.8)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**方法B: 使用Conda安装**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 步骤5: 安装项目依赖

```bash
pip install pyvista pandas numpy geopandas pvlib-python pyproj pyyaml tqdm trimesh
```

---

### 🐧 Linux 环境配置

#### 步骤1: 检查NVIDIA驱动

```bash
nvidia-smi
```

**如果未安装驱动:**

**Ubuntu/Debian:**
```bash
# 方法1: 使用ubuntu-drivers (推荐)
sudo ubuntu-drivers autoinstall

# 方法2: 手动安装
sudo apt update
sudo apt install nvidia-driver-525  # 或其他版本号

# 重启系统
sudo reboot

# 验证安装
nvidia-smi
```

**CentOS/RHEL:**
```bash
# 添加NVIDIA仓库
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# 安装驱动
sudo dnf install nvidia-driver

# 重启系统
sudo reboot

# 验证安装
nvidia-smi
```

**从源代码安装 (通用方法):**
```bash
# 下载驱动
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.60.13/NVIDIA-Linux-x86_64-525.60.13.run

# 安装依赖
sudo apt install build-essential  # Ubuntu
# 或
sudo yum groupinstall "Development Tools"  # CentOS

# 安装驱动
sudo bash NVIDIA-Linux-x86_64-525.60.13.run

# 重启
sudo reboot
```

#### 步骤2: 安装CUDA Toolkit (可选但推荐)

**Ubuntu:**
```bash
# 添加CUDA仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# 或者安装特定版本 (推荐CUDA 11.8)
sudo apt-get install cuda-11-8
```

**设置环境变量:**
```bash
# 编辑 ~/.bashrc 或 ~/.zshrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
```

#### 步骤3: 创建Python虚拟环境

**使用venv:**
```bash
# 安装venv (如果未安装)
sudo apt install python3-venv python3-pip  # Ubuntu
# 或
sudo yum install python3-venv python3-pip  # CentOS

# 创建虚拟环境
python3 -m venv ~/pv_gpu_env
source ~/pv_gpu_env/bin/activate
```

**使用Conda (推荐):**
```bash
# 下载并安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建环境
conda create -n pv_gpu python=3.9
conda activate pv_gpu
```

#### 步骤4: 安装PyTorch (CUDA版本)

```bash
# 使用pip (推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或使用conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 步骤5: 安装项目依赖

```bash
# 更新pip
pip install --upgrade pip

# 安装依赖
pip install pyvista pandas numpy geopandas pvlib-python pyproj pyyaml tqdm trimesh

# 如果遇到编译错误，安装开发工具
sudo apt install build-essential python3-dev  # Ubuntu
# 或
sudo yum groupinstall "Development Tools" python3-devel  # CentOS
```

---

## 🧪 验证GPU安装 (通用)

### 创建测试脚本

创建 `test_gpu.py` 文件:

```python
import torch
import sys

print("="*60)
print("PyTorch GPU环境检测")
print("="*60)

# PyTorch版本
print(f"\nPyTorch版本: {torch.__version__}")

# CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA可用: {cuda_available}")

if cuda_available:
    # CUDA版本
    print(f"CUDA版本: {torch.version.cuda}")

    # GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")

    # GPU信息
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  总显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")

    # 测试GPU计算
    print("\n测试GPU计算...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU计算测试成功")
    except Exception as e:
        print(f"❌ GPU计算测试失败: {e}")
else:
    print("\n⚠️ CUDA不可用，将使用CPU模式")
    print("   如果您有NVIDIA GPU，请检查:")
    print("   1. NVIDIA驱动是否正确安装")
    print("   2. PyTorch是否安装了CUDA版本")
    print("   3. 使用命令: pip install torch --index-url https://download.pytorch.org/whl/cu118")

print("="*60)
```

运行测试:

```bash
python test_gpu.py
```

**期望输出 (GPU可用):**
```
============================================================
PyTorch GPU环境检测
============================================================

PyTorch版本: 2.1.0+cu118
CUDA可用: True
CUDA版本: 11.8
GPU数量: 1

GPU 0:
  名称: NVIDIA GeForce RTX 3080
  总显存: 10.0 GB
  计算能力: 8.6

测试GPU计算...
✅ GPU计算测试成功
============================================================
```

---

## 🔧 配置脚本使用GPU

### 方法1: 在配置文件中启用

编辑 `config.yaml`:

```yaml
computation:
  use_gpu: true        # 启用GPU
  batch_size: 100      # 批处理大小 (根据显存调整)
```

### 方法2: 命令行参数

```bash
# 启用GPU (默认)
python main_pv_calculation_gpu.py --config config.yaml

# 禁用GPU (强制使用CPU)
python main_pv_calculation_gpu.py --config config.yaml --no-gpu
```

---

## ⚙️ 性能调优

### 调整批处理大小 (batch_size)

根据显存大小调整 `batch_size`:

| 显存 | 推荐batch_size | 说明 |
|-----|---------------|------|
| 4GB | 20-50 | 小批量处理 |
| 6GB | 50-100 | 中等批量 |
| 8GB | 100-200 | 大批量 |
| 12GB+ | 200-500 | 超大批量 |

**如果遇到 "CUDA out of memory" 错误:**

```yaml
computation:
  batch_size: 50  # 减小批处理大小
```

### 监控显存使用

**Windows (PowerShell):**

```powershell
# 持续监控
while($true) {
    Clear-Host
    nvidia-smi
    Start-Sleep -Seconds 1
}
```

**Linux/Windows (新终端):**
```bash
# Linux
watch -n 1 nvidia-smi

# Windows Git Bash
while true; do clear; nvidia-smi; sleep 1; done
```

---

## 🐛 常见问题排查

### 问题1: "CUDA not available"

**可能原因:**
1. 未安装NVIDIA驱动
2. 安装了CPU版本的PyTorch
3. CUDA版本不匹配

**解决方案 (Windows):**
```bash
# 卸载旧版本
pip uninstall torch torchvision torchaudio

# 重新安装CUDA版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**解决方案 (Linux):**
```bash
# 卸载旧版本
pip uninstall torch torchvision torchaudio

# 重新安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果仍然不行，检查LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 问题2: "CUDA out of memory"

**解决方案:**
1. 减小 `batch_size`
2. 简化建筑mesh (减少三角形数)
3. 分批处理轨迹数据

```yaml
# 修改 config.yaml
computation:
  batch_size: 50  # 从100减小到50
```

```python
# 在脚本中手动释放显存
import torch
torch.cuda.empty_cache()
```

### 问题3: Linux下找不到libcudart.so

**解决方案:**
```bash
# 查找CUDA库路径
find /usr -name "libcudart.so*" 2>/dev/null

# 添加到LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 永久添加到 ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 问题4: Windows下GPU利用率很低

**可能原因:**
- 批处理大小太小
- 数据传输成为瓶颈
- 电源管理限制

**解决方案:**
```bash
# 设置Windows高性能模式
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# 增大batch_size
# 修改 config.yaml
computation:
  batch_size: 200  # 根据显存调整
```

### 问题5: Linux权限问题

**解决方案:**
```bash
# 添加用户到video组
sudo usermod -a -G video $USER

# 重新登录或执行
newgrp video

# 验证
groups
```

---

## 📊 性能基准测试 (通用)

```python
# benchmark_gpu.py
import time
import torch
import numpy as np

def benchmark_gpu_vs_cpu():
    print("GPU vs CPU 性能测试")
    print("="*60)

    # 测试参数
    n_rays = 1000000  # 100万条光线

    # CPU测试
    print("\nCPU测试...")
    origins = np.random.randn(n_rays, 3).astype(np.float32)
    directions = np.random.randn(n_rays, 3).astype(np.float32)

    start = time.time()
    result_cpu = origins - directions * 5e5
    cpu_time = time.time() - start
    print(f"  耗时: {cpu_time:.3f} 秒")

    # GPU测试
    if torch.cuda.is_available():
        print("\nGPU测试...")
        origins_gpu = torch.from_numpy(origins).cuda()
        directions_gpu = torch.from_numpy(directions).cuda()

        torch.cuda.synchronize()
        start = time.time()
        result_gpu = origins_gpu - directions_gpu * 5e5
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  耗时: {gpu_time:.3f} 秒")

        speedup = cpu_time / gpu_time
        print(f"\n加速比: {speedup:.1f}x")
    else:
        print("\nGPU不可用，跳过测试")

    print("="*60)

if __name__ == "__main__":
    benchmark_gpu_vs_cpu()
```

运行:
```bash
python benchmark_gpu.py
```

---

## 📝 快速配置命令 (复制粘贴)

### Windows 快速配置

```bash
# 1. 检查NVIDIA驱动
nvidia-smi

# 2. 创建conda环境
conda create -n pv_gpu python=3.9 -y
conda activate pv_gpu

# 3. 安装PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安装项目依赖
pip install pyvista pandas numpy geopandas pvlib-python pyproj pyyaml tqdm trimesh

# 5. 验证GPU
python test_gpu.py
```

### Linux 快速配置 (Ubuntu/Debian)

```bash
# 1. 安装NVIDIA驱动
sudo ubuntu-drivers autoinstall
sudo reboot

# 2. 验证驱动
nvidia-smi

# 3. 创建虚拟环境
conda create -n pv_gpu python=3.9 -y
conda activate pv_gpu

# 4. 安装PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 安装项目依赖
pip install pyvista pandas numpy geopandas pvlib-python pyproj pyyaml tqdm trimesh

# 6. 验证GPU
python test_gpu.py
```

### Linux 快速配置 (CentOS/RHEL)

```bash
# 1. 安装NVIDIA驱动
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf install nvidia-driver -y
sudo reboot

# 2. 验证驱动
nvidia-smi

# 3. 创建虚拟环境
python3 -m venv ~/pv_gpu_env
source ~/pv_gpu_env/bin/activate

# 4. 安装PyTorch
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 安装项目依赖
pip install pyvista pandas numpy geopandas pvlib-python pyproj pyyaml tqdm trimesh

# 6. 验证GPU
python test_gpu.py
```

---

## 🎯 推荐配置

### 最佳性能配置 (RTX 3080, 10GB显存)

```yaml
computation:
  use_gpu: true
  batch_size: 200
  time_resolution_minutes: 1
```

### 低显存配置 (GTX 1060, 6GB显存)

```yaml
computation:
  use_gpu: true
  batch_size: 50
  time_resolution_minutes: 5  # 降低时间分辨率
```

### CPU备用配置 (无GPU)

```yaml
computation:
  use_gpu: false
  batch_size: 10
  time_resolution_minutes: 5
```

---

## ✅ 总结

### GPU加速的优势

- ✅ 计算速度提升 8-10倍
- ✅ 支持更高时间分辨率 (1分钟)
- ✅ 可处理更大规模数据

### GPU加速的要求

- ✅ NVIDIA GPU (支持CUDA)
- ✅ 正确的驱动和PyTorch版本
- ✅ 足够的显存

### 如果没有GPU

- ✅ 脚本会自动回退到CPU模式
- ✅ 所有功能依然可用
- ⚠️ 计算速度较慢，建议降低时间分辨率

---

## 📞 故障排查检查清单

遇到问题时，按顺序检查：

**1. 检查NVIDIA驱动**

```bash
nvidia-smi
```

**2. 检查CUDA是否可用**

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

**3. 检查PyTorch版本**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

**4. 检查CUDA版本匹配**

```bash
python -c "import torch; print(f'CUDA版本: {torch.version.cuda}')"
```

**5. Linux特有: 检查库路径**

```bash
echo $LD_LIBRARY_PATH
ldconfig -p | grep cuda
```

---

## 🆘 获取帮助

遇到问题时，请提供以下信息：

1. **系统信息**
   - 操作系统: Windows/Linux版本
   - GPU型号: `nvidia-smi` 输出

2. **软件信息**
   - Python版本: `python --version`
   - PyTorch版本: `python test_gpu.py` 输出

3. **完整错误信息**
   - 截图或复制完整的错误堆栈

---

## 📚 相关资源

- **PyTorch安装**: https://pytorch.org/get-started/locally/
- **NVIDIA驱动下载**: https://www.nvidia.com/Download/index.aspx
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
- **项目文档**: 参见 `README.md`

---

祝您使用顺利！🚀

如有问题，欢迎提Issue或联系项目维护者。
