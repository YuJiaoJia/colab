import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from u2net.src import u2net_lite,u2net_litte,u2net_lite_half_m32,u2net_lite_half_m64,u2net_lite_half_m48
from u2net.src.rep_half_u2net import rep_u2net_lite_half_m48
from train_utils.train_and_eval import evaluate


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def evaluate_res(model,val_data_loader, device):
    mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
    mae_info = mae_metric.compute()
    f1_info, precision, recall = f1_metric.compute()


def main():
    weights_path = "save_weights/u2net_lite_half_48m_best/model_best.pth"
    img_path = "./train/train_img/300100/001001.bmp"

    threshold = 0.5

    assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(320),
        transforms.Normalize(mean=(0.5,),
                             std=(0.5,))
    ])

    origin_img = cv2.imread(img_path, 0)

    h, w = origin_img.shape[:2]
    img = data_transform(origin_img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

    model =u2net_lite_half_m48()
    weights = torch.load(weights_path, map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()#将模型切换到评估模式

    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 1, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        pred = model(img)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]

        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask = np.where(pred > threshold, 1, 0)
        #origin_img = np.array(origin_img, dtype=np.uint8)
        # seg_img = origin_img * pred_mask#本来pred_mask[..., None]，由于传入图像是灰度图，所以不需要再加一维
        plt.imshow(pred_mask,'gray')
        plt.show()
        #cv2.imwrite("pred_result.png", cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))



        return pred


if __name__ == '__main__':
    print("start predict")
    main()