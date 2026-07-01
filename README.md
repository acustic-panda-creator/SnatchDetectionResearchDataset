# Multi-Modal Snatch Detection System

This project is an advanced, multi-modal machine learning pipeline designed to detect snatching incidents (e.g., chain/bag snatching) using video and audio data. It tracks entities, analyzes motion physics, evaluates interactions, and fuses video features with audio anomalies to produce highly accurate alerts.

---

## 📁 Repository Structure & File Descriptions

Based on the project structure, here is a breakdown of the repository:

*   **`ModelTraining/`**: Contains the development and training notebooks.
    *   **`Audio Model/`**: 
        *   `AudioModelTrain.ipynb` & `Audio_project_Traning_GRUBa...`: Notebooks used to extract Mel Spectrograms and train the CRNN-GRU audio anomaly model.
    *   **`Video Model/`**: 
        *   `XGBoost_Model_Train.ipynb`: Notebook for training the core Machine Learning model on extracted features.
        *   `YoloDetection_feature_extracti...`: Notebook for testing YOLOv8 tracking and spatial feature extraction.
        *   `validation_debugger.ipynb`: Script for validating frame-by-frame tracker accuracy.
        *   `yolov8n.pt`: The base YOLOv8 nano model weights used for entity detection.
*   **`Project Files/`**: Contains the final execution pipelines.
    *   **`MultiModel_SnatchDetection.ipynb`**: The **Hybrid Model**. Runs both Audio and Video inference simultaneously and fuses the results. Use this for the most accurate predictions.
    *   **`Video_SnatchDetection.ipynb`**: The **Video-Only Model**. Bypasses audio extraction entirely. Use this if your video has no sound or if you want to test purely spatial/physics-based detection.
*   **`Videos/`**: A directory intended to store your testing footage.
*   **`audio_crnn_gru_model.h5`**: The compiled Keras model for audio classification.
*   **`snatch_model.pkl`**: The compiled XGBoost model for video feature classification.

---

## How to Run on Google Colab

### Step 1: Upload Required Files
Before running the code, open your Colab file explorer (folder icon on the left) and upload the following files to the base `/content/` directory:
1.  `snatch_model.pkl` (Required for both pipelines)
2.  `audio_crnn_gru_model.h5` (Required ONLY for the Multi-Modal pipeline)
3.  `snatch_detection_requirements.txt` (Required for setup)
4.  Your test video (e.g., `Shoot_7.mp4`)

### Step 2: Install Dependencies
Run the following command in your first Colab cell to ensure exact version compatibility:
```bash
!pip install -r snatch_detection_requirements.txt
```
### Step 3: Choose Your Pipeline
For Video + Audio (Hybrid): Open the code from MultiModel_SnatchDetection.ipynb. Ensure INPUT_VIDEO_PATH, VIDEO_MODEL_PATH, and AUDIO_MODEL_PATH point to your uploaded files.
For Video-Only: Open the code from Video_SnatchDetection.ipynb. Ensure INPUT_VIDEO_PATH and ML_MODEL_PATH point to your uploaded files. (No audio model is needed).

### Step 4: Execute
Run the pipeline block. The output video will be generated and saved as Final_Snatch_Output.mp4 in the Colab file explorer. You can download it by clicking the three dots next to the file.
