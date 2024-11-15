### Note: You need to install pytorch video and pytorch in advance (depending on your CUDA version)

##### step1：env:
```
conda create -n WVLED python=3.8
conda activate WVLED
```

##### step2: pytorchvideo:
```
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .
```

##### step3:
`pip install -r requirements.txt`

##### How to training:
`python train.py --cfg configs/UCF101_24_base.yaml`

&ensp;Note: You must modify the classes in configs/defaults.py to the number of classes in th dataset. eg：_C.MODEL_PARA.CLASSES = 24（for UCF101-24）

##### The data location: You need to put the dataset in the "data" folder in the following format（eg. UCF101-24）:
```
UCF101-24
|_ annotations
|  |_ ucf101_24_action_list.pbtxt
|_ Frames
|  |_ [video_file1]
|  |_ [video_file2]
|_ UCF101v2-GT.pkl
```
##### The ucf101_24_action_list.pbtxt format:
```
item {
  name: "Basketball"
  id: 0
}
item {
  name: "BasketballDunk"
  id: 1
}
...
item {
  name: "WalkingWithDog"
  id: 23
}
```
