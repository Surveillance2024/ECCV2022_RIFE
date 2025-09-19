import os
import sys
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
from io import BytesIO
import numpy as np

INTERPOLATOR_ROOT = os.path.dirname(__file__)  # INTERPOLATOR_ROOT root directory
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
if str(INTERPOLATOR_ROOT) not in sys.path:
    sys.path.append(str(INTERPOLATOR_ROOT))  # add INTERPOLATOR_ROOT to PATH

class InterpolatorInterface:
    def __init__(self, model_pth=os.path.join(INTERPOLATOR_ROOT, "train_log") ): # os.path.join(INTERPOLATOR_ROOT, "RIFEv4.22", "train_log")):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        try:
            try:
                try:
                    from model.RIFE_HDv2 import Model
                    self.model = Model()
                    self.model.load_model(model_pth, -1)
                    print("Loaded v2.x HD model.")
                except:
                    from train_log.RIFE_HDv3 import Model
                    self.model = Model()
                    self.model.load_model(model_pth, -1)
                    print("Loaded v3.x HD model.")
            except:
                from model.RIFE_HD import Model
                self.model = Model()
                self.model.load_model(model_pth, -1)
                print("Loaded v1.x HD model")
        except:
            from model.RIFE import Model
            self.model = Model()
            self.model.load_model(model_pth, -1)
            print("Loaded ArXiv-RIFE model")
        self.model.eval()
        # self.model.to()
        self.model.device()
    def generate(
        self,
        imgs: tuple[str | np.ndarray | bytes, str | np.ndarray | bytes] | None = None,
        exp: int = 4,
        ratio: float = 0,
        rthreshold: float = 0.02,
        rmaxcycles: int = 8,
        outputdir: str | None = None
    ) -> list[str] | list[np.ndarray]:
        """
        imgs can be:
          - tuple of file paths: (str, str)
          - tuple of numpy arrays: (np.ndarray, np.ndarray)
          - tuple of raw image bytes: (bytes, bytes)
          - or any mix (str, np.ndarray, bytes)
        """

        def _load_img(img):
            if isinstance(img, str):  # file path
                if not os.path.isfile(img):
                    raise FileNotFoundError(f"Image file not found: {img}")
                if img.endswith('.exr'):
                    p_img = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
                    p_img = (torch.tensor(p_img.transpose(2, 0, 1)).to(self.device)).unsqueeze(0)
                    return p_img
                else:
                    p_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                    p_img = (torch.tensor(p_img.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
                    return p_img
            elif isinstance(img, np.ndarray): # must br in BGR format
                p_img = (torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).to(self.device) / 255.).unsqueeze(0)
                return p_img
            elif isinstance(img, (bytes, bytearray, BytesIO)):  # raw image bytes
                if isinstance(img, BytesIO):
                    img = img.getvalue()
                arr = np.frombuffer(img, np.uint8)
                p_img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if p_img is None:
                    raise ValueError(f"Failed to decode image bytes: {img}")
                p_img = (torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).to(self.device) / 255.).unsqueeze(0)
                return p_img
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

        img0 = _load_img(imgs[0])
        img1 = _load_img(imgs[1])

        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)


        if ratio:
            img_list = [img0]
            img0_ratio = 0.0
            img1_ratio = 1.0
            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(rmaxcycles):
                    middle = self.model.inference(tmp_img0, tmp_img1)
                    middle_ratio = ( img0_ratio + img1_ratio ) / 2
                    if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                        break
                    if ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
        else:
            img_list = [img0, img1]
            for i in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        if outputdir is not None:
            def get_next_index(directory, prefix):
                existing_files = [f for f in os.listdir(directory) if f.startswith(prefix)]
                indices = []
                for f in existing_files:
                    basename = os.path.basename(f)
                    name, ext = os.path.splitext(basename)
                    if name[len(prefix):].isdigit():
                        indices.append(int(name[len(prefix):]))
                return max(indices, default=-1) + 1
            

            if not os.path.exists(outputdir):
                os.mkdir(outputdir)

            next_index = get_next_index(outputdir, 'img')
            
            output_list = []

            for i in range(len(img_list)):
                if imgs[0].endswith('.exr') and imgs[1].endswith('.exr'):
                    new_img_pth = os.path.join(outputdir, 'img{}.exr'.format(next_index+i))
                    cv2.imwrite(new_img_pth, (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                    output_list.append(new_img_pth)
                else:
                    new_img_pth = os.path.join(outputdir, 'img{}.png'.format(next_index+i))
                    cv2.imwrite(new_img_pth, (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
                    output_list.append(new_img_pth)
            return output_list
        else:
            return_list = []
            for i in range(len(img_list)):
                if imgs[0].endswith('.exr') and imgs[1].endswith('.exr'):
                    return_list.append((img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w])
                else:
                    return_list.append((img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
            return return_list
