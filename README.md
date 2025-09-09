# dnn-gpu-training-tinyimagenet
Deep Neural Network Training &amp; Profiling with TensorFlow on NVIDIA GPUs (TinyImageNet)
# Deep Neural Network Training on NVIDIA GPU (TinyImageNet)

This project demonstrates hardware-accelerated training and inference of a Convolutional Neural Network (CNN) on the **TinyImageNet dataset** using **TensorFlow** and **NVIDIA GPUs** accessed via SSH tunneling.  

### 🚀 Highlights
- Implemented and profiled a CNN model (`CNN_tinyimagenet.h5`) in **TensorFlow**.  
- Trained and validated the model with different **batch sizes** and **epoch counts** to measure their impact on accuracy.  
- Explored **inference profiling** using TensorBoard:
  - Single image inference latency  
  - Online inference (10, 100, 1000 images)  
  - Batch inference (batch sizes 20–200)  
- Leveraged **NVIDIA GPU acceleration** via remote **SSH tunneling** to reduce training time.  
- Explored model internals with **Netron visualization** (weights, kernels, feature maps).  

### 📊 Key Results
- Achieved strong Top-1 and Top-5 validation accuracy across experiments.  
- Observed how GPU acceleration significantly improved throughput compared to CPU-only execution.  
- Showcased the trade-offs between **batch inference** vs **online inference** for real workloads.  

### 🔧 Tech Stack
- **TensorFlow 2.16** with CUDA support  
- **Jupyter Notebook** (remote execution via SSH tunnel)  
- **TensorBoard** for profiling  
- **Netron** for model visualization  

### 📂 Repository Structure
├── lab1_notebook.ipynb # Jupyter Notebook (code, outputs, profiling results)
├── lab1_pdf
├─  Binary Files


### 💡 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/dnn-gpu-training-tinyimagenet.git
   cd dnn-gpu-training-tinyimagenet
python3 -m venv lab1_venv
source lab1_venv/bin/activate
pip install -r requirements.txt
jupyter notebook
ssh -X <netid>@<gpu-vm-address> -L 5050:localhost:5050
jupyter notebook --allow-root --no-browser --port 5050

