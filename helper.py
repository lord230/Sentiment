import os
import subprocess
import shutil
import sys

def install_modules(modules):
    """
    This Function accepts Modules in a list form so pass a LIST

    """
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            print(f"Installing {module}...")
            subprocess.check_call(["uv", "pip", "install", module])

def create_structure(structure):

    """
    This function creates directories 
    
    """
    dataset_dirs = []  

    for root, sub_dirs, files in structure:
        if not os.path.exists(root):
            os.makedirs(root)
            print(f"Created directory: {root}")
        for sub_dir in sub_dirs:
            path = os.path.join(root, sub_dir)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created sub-directory: {path}")

            
            if sub_dir == "dataset":
                dataset_dirs.append(path)

        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("# Created By LORD \n")
                print(f"Created file: {file_path}")
    
    return dataset_dirs  # Return dataset directories

def copy_dataset(source_path, target_dirs):
    """
    Copies the dataset from the source path to multiple target directories.
    Works on both Windows and macOS/Linux.
    """
    for target_dir in target_dirs:
        if os.path.exists(source_path):
            try:
                if not os.listdir(target_dir):  
                    shutil.copytree(source_path, target_dir, dirs_exist_ok=True)
                    print(f"Copied dataset to: {target_dir}")
                else:
                    print(f"Skipped copying to {target_dir} (already contains files).")
            except Exception as e:
                print(f"Error copying dataset to {target_dir}: {e}")
        else:
            print(f"Source dataset path {source_path} does not exist!")


def download_dataset(link , dataset_dirs):
    import kagglehub

    if input("Do you want to download the dataset? (yes/no): ").strip().lower() == "yes":
        dataset_path = kagglehub.dataset_download(link)
        print(f"Dataset downloaded at: {dataset_path}")
        copy_dataset(dataset_path, dataset_dirs)


if __name__ == "__main__":
    required_modules = ["numpy", "pandas", "requests", "torch", "torchvision", "kagglehub"] 
    
    project_structure = [
        ("Transformers", ["Models", "CheckPoints", "results", "runs", "dataset"],
         ["model.py", "train.py", "__init__.py", "utils.py", "config.py", "dataset.py"]),
        ("Transformers/Models", [], ["__init__.py", "blocks.py", "Stocks_M.py","Product_M.py","Twitter_M.py"]),
    ]
    
    install_modules(required_modules)
    link = "pankrzysiu/cifar10-python"
    dataset_dirs = create_structure(project_structure)
    # download_dataset(link , dataset_dirs)
    print("Setup complete!")
