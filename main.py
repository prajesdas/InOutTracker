
from conter import *
from showClassInModel import *
from ultralytics import YOLO

def main():
    video=r"E:\Count-entering-and-exiting-people-using-YOLOv8-main-main\p.mp4"
    model=r"E:\Count-entering-and-exiting-people-using-YOLOv8-main-main\yolov8s.pt"
    
    model=YOLO(model)
    # showClass(model.names)
    # showDatainFile()
    
    conter=Conter(video,model)
    conter()

if __name__ == "__main__":
    main()