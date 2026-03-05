# Project Setup Helper 

This repository contains a **Python helper script** that automatically:

- Installs required Python modules  
- Creates a predefined project structure (ResNet, AlexNet, VGG)  
- Downloads and copies datasets via [kagglehub](https://pypi.org/project/kagglehub/)  

It helps you quickly **bootstrap deep learning projects** with an organized structure.

---

##  Project Structure

The script will generate the following structure:

Resnet/
│── Models/
│ ├── init.py
│ ├── blocks.py
│ └── resnet.py
│── CheckPoints/
│── results/
│── runs/
│── dataset/
│── model.py
│── train.py
│── init.py
│── utils.py
│── config.py
│── dataset.py

alex_net/
│── Models/
│ ├── init.py
│ ├── blocks.py
│ └── alexnet.py
│── CheckPoints/
│── results/
│── runs/
│── dataset/
│── model.py
│── train.py
│── init.py
│── utils.py
│── config.py
│── dataset.py

vgg_exp/
│── Models/
│ ├── init.py
│ ├── blocks.py
│ └── vgg.py
│── CheckPoints/
│── results/
│── runs/
│── dataset/
│── model.py
│── train.py
│── init.py
│── utils.py
│── config.py
│── dataset.py



---

##  Features

- **Auto Install Dependencies**  
  Installs required modules if missing (`numpy`, `pandas`, `requests`, `torch`, `torchvision`, `kagglehub`).

- **Project Structure Setup**  
  Creates organized folders and starter `.py` files.

- **Dataset Handling**  
  - Downloads dataset from KaggleHub (default: CIFAR-10).  
  - Copies dataset into all experiment `dataset/` folders.  
  - Skips copying if files already exist.

---

## 🔧 Usage

### Clone Repository
```bash
  git clone https://github.com/lord230/helper.git
  cd helper
  
  2. Run the Helper Script
  python setup_helper.py

