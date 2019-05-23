## 1. Extract frames:
* script: /home/kskripka/Krossover/train/tools/FrameExtractor/FrameExtractor.sh
* parameters:
	* -file - path to the input file
	*-start - time (in seconds) to start making frames
	*-end(optional) - time (in seconds) to end making frames
	*-out - folder where to place extracted frames. Folder must be created manually
	*-step - parameter that indicates frames step
* example: python3 FrameExtractor.py -file="/home/kskripka/Krossover/video/Game_140176.mp4" -start=0 -end=60 -out="/home/kskripka/Krossover/video/frames/Game_140176" -step=5

## 2. Prepare images and region data:
* Use Via (html image region annotation tool): http://www.robots.ox.ac.uk/~vgg/software/via/
* Open project file (via_basketball_train.json or via_basketball_val.json)
* Make sure project have correct path to images specified in settings (note the trailing / and \)
* When regions are finished, export annotations as json and save to corresponding folder (train or val). Regin file name must be via_region_data.json

## 3. Run training:
* script: /home/kskripka/Krossover/train/mrcnn/train_basketball.sh
* parameters:
	* --dataset - path to dataset where train and val folders are located
	* --weights - path to the h5 weights file to start training with (either mask_rcnn_coco.h5 or pre-trained mask_rcnn_basketball.h5)
	* --epoch - epoch count for training. The more epoch is used the longer training. If you train from coco model, use 50-100 epochs. For pre-trained model with our objects use about 30 epochs)
* example: python3 basketball/basketball.py train --dataset=/home/kskripka/Krossover/train/mrcnn/basketball/ --weights=/home/kskripka/Krossover/train/logs/22.05.2019_100_30_mask_rcnn_basketball.h5 --epoch=50

## 4. Run detection on video:
* script: /home/kskripka/Krossover/mask_weight.sh
* parameters:
	* -i - path to the input video file. Use short video files for testing (2 mins is enough)
	* -f - frame step to run detection
	* -mode - output mode: 1 = csv, 2 = video, 3 = both csv and video
	* -o - output folder
	* -w - path to the weight file to use for detection
	* -c - min detection confidence (default: 0.7 = 70%, all other detections will be ignored)
* example: python3 mask_weight.py -i "/home/kskripka/Krossover/video/NBA_20.05.2019.50fps.mp4" -f 5 -mode 3 -o /home/kskripka/Krossover/out/trained_mrcnn -w /home/kskripka/Krossover/train/logs/mask_rcnn_basketball.h5 -c 0.7
