
#=========输入提示词,来获得maks图片.
pat='/mnt/e/sam_vit_b_01ec64.pth'

promp=''
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint=pat)
mask_generator = SamAutomaticMaskGenerator(sam)

import cv2
image = cv2.imread('demo.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



masks = mask_generator.generate(image)
print(masks)