from setuptools import setup, find_packages

setup(
    name="yolov8-yolox-style",
    version="8.0.0",
    description="YOLOv8 (Ultralytics, 2023) in YOLOX-official project structure",
    packages=find_packages(),
    install_requires=["torch>=1.9.0", "numpy", "opencv-python", "pyyaml", "tqdm"],
    python_requires=">=3.8",
)
