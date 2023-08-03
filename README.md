
# Dog Detection in Video

此程式使用OpenCV和YOLOv3在影片中檢測和標記狗的位置。結果將以圖形視覺化方式保存為新的影片，並以CSV文件的形式記錄檢測到狗的時間戳。

## 需求

- Python 3.6或更高版本
- OpenCV 4.0或更高版本
- NumPy
- Pandas

您可以使用以下指令安裝需要的套件：

```bash
pip install opencv-python numpy pandas
```

## 使用方式

1. 下載YOLOv3的權重文件和配置文件。您需要下載"YOLOv3-416"的權重(weights)和配置文件(cfg)，這是一種適合大多數場景和電腦的通用模型。您可以在以下網址找到這些文件：https://pjreddie.com/darknet/yolo/
2. 下載coco.names標籤文件，這是一個包含80個類別的標籤文件，包括"dog"。您可以在以下網址找到這個文件：https://github.com/pjreddie/darknet/blob/master/data/coco.names
3. 將您要分析的影片保存為文件。
4. 修改程式碼中的路徑，使其指向您的權重、配置、標籤和影片文件。
5. 選擇輸出影片和CSV文件的路徑。
6. 在終端機或命令提示字元中運行程式：

   ```bash
   python your_script.py
   ```

   請將`your_script.py`替換為您存放程式碼的文件名。

## 函數說明

`detect_dogs_in_video(video_path, weights_path, config_path, output_video_path, output_csv_path)`

- `video_path`：要處理的影片的路徑。
- `weights_path`：YOLOv3權重文件的路徑。
- `config_path`：YOLOv3配置文件的路徑。
- `output_video_path`：輸出影片的路徑。
- `output_csv_path`：輸出CSV文件的路徑，其中包括檢測到狗的時間戳。

## Get Started

下面是一個基本的"開始使用"指南：

1. 首先，確保您的系統已經安裝了Python 3.6或更高版本。您可以在終端機或命令提示字元中輸入`python --version`來檢查您的Python版本。
2. 使用pip來安裝需要的Python套件。在終端機或命令提示字元中輸入上述的指令。
3. 下載需要的YOLOv3文件和coco.names標籤文件。請確保您已經有一個要分析的影片文件。
4. 打開Python程式碼文件，將路徑修改為對應您的文件位置。選擇輸出影片和CSV文件的路徑。
5. 儲存並關閉程式碼文件。在終端機或命令提示字元中，切換到您的程式碼文件所在的目錄，並運行程式。
6. 程式將處理影片，並將結果保存為新的影片和CSV文件。您可以在輸出的影片中看到檢測到的狗的位置，並在CSV文件中查看檢測到狗的時間戳。
