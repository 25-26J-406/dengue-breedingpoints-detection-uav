# Detecting Dengue Breeding Points Using UAV Images 🦟🚁

## 📌 Overview
This research focuses on detecting potential dengue mosquito breeding points using UAV (drone) images.  
By applying deep learning–based object detection (YOLOv8), the system aims to identify stagnant water sources and high-risk areas from aerial imagery to support early intervention and prevention.

## 🎯 Research Objectives
- Identify dengue breeding-prone locations from UAV images
- Apply YOLOv8 for accurate object detection
- Reduce manual field inspections using automated aerial analysis
- Support public health authorities with data-driven insights

## 🧠 Technologies & Tools
- **Model:** YOLOv8
- **Domain:** Computer Vision, Object Detection
- **Input:** UAV (Drone) Images
- **Frameworks/Libraries:** PyTorch, Ultralytics YOLO
- **Environment:** Google Colab / Local Machine

## 🗂️ Dataset
- UAV images collected from dengue-risk environments
- Images annotated in YOLO format
- Includes both breeding and non-breeding locations

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│        Mosquito Breeding Site Detection System                  │
└─────────────────────────────────────────────────────────────────┘


     ┌──────────────────────────────────────────────────┐
     │            INPUT LAYER                           │
     ├──────────────────────────────────────────────────┤
     │                                                  │
     │  • Drone/UAV Images                              │
     │  • Satellite Imagery                             │
     │  • Manual Uploads                                │
     │                                                  │
     └────────────────────┬─────────────────────────────┘
                          │
                          ▼
     ┌──────────────────────────────────────────────────┐
     │         IMAGE PREPROCESSING                      │
     ├──────────────────────────────────────────────────┤
     │                                                  │
     │  • Image Enhancement                             │
     │  • Resize & Normalize                            │
     │  • Data Augmentation                             │
     │                                                  │
     └────────────────────┬─────────────────────────────┘
                          │
                          ▼
     ┌──────────────────────────────────────────────────┐
     │         DETECTION MODEL                          │
     ├──────────────────────────────────────────────────┤
     │                                                  │
     │          YOLOv8 / CNN Model                      │
     │                                                  │
     │  Detects:                                        │
     │  • Stagnant Water                                │
     │  • Water Containers                              │
     │  • Tires, Buckets, Pots                          │
     │  • Drains & Gutters                              │
     │                                                  │
     └────────────────────┬─────────────────────────────┘
                          │
                          ▼
     ┌──────────────────────────────────────────────────┐
     │         POST-PROCESSING                          │
     ├──────────────────────────────────────────────────┤
     │                                                  │
     │  • Filter Results (Confidence > 0.5)             │
     │  • GPS Coordinate Mapping                        │
     │  • Risk Level Assessment                         │
     │                                                  │
     └────────────────────┬─────────────────────────────┘
                          │
                          ▼
     ┌──────────────────────────────────────────────────┐
     │         DATABASE STORAGE                         │
     ├──────────────────────────────────────────────────┤
     │                                                  │
     │  • Detection Records                             │
     │  • GPS Coordinates                               │
     │  • Timestamps                                    │
     │  • Risk Scores                                   │
     │                                                  │
     └────────────────────┬─────────────────────────────┘
                          │
                          ▼
     ┌──────────────────────────────────────────────────┐
     │         OUTPUT LAYER                             │
     ├──────────────────────────────────────────────────┤
     │                                                  │
     │  • Admin Dashboard (Heat Maps)                   │
     │  • Mobile App (Field Teams)                      │
     │  • Alert System (PHI Notifications)              │
     │                                                  │
     └──────────────────────────────────────────────────┘
```
## 🔮 Future Improvements
- Planned dataset enhancement to improve model robustness and accuracy  
- Support for large-size UAV images using cropping and overlapping techniques  
- Ongoing YOLOv8 model fine-tuning for improved detection performance  
- Generation of summary reports for detected areas (location, time, risk level)  
- UI and visualization improvements scheduled for the next development phase  

## 👤 Author
**IT22582010**  
**Piyasena S H T**  
Detection of Mosquito Breeding Sites (Aerial) 

## 📄 License
This project is for academic and research purposes.
