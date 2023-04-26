
#=========输入提示词,来获得maks图片.
pat='/mnt/e/sam_vit_h_4b8939.pth'

promp=''
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["default"](checkpoint=pat)
predictor = SamPredictor(sam)
predictor.set_image('demo.jpg')
masks, _, _ = predictor.predict('')