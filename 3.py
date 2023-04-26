
#=========输入提示词,来获得maks图片.
pat='/mnt/e/sam_vit_b_01ec64.pth'   #下面我们使用小权重来debug代码. 当前要自己手动下载权重到e盘.

promp=''
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint=pat)
predictor = SamPredictor(sam)

import cv2
image = cv2.imread('demo.jpg')
predictor.set_image(image)
masks, _, _ = predictor.predict('')