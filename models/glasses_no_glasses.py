import torch
import os
import cv2

# The model is just 3Mb
if torch.cuda.is_available():
    MODEL_PATH = "/Users/einstalek/Downloads/best_model_ckpt.pt"
else:
    MODEL_PATH = "/Users/einstalek/Downloads/best_model_ckpt_cpu.pt"

IMG_DIRS = ["/Users/einstalek/Downloads/with_glasses",
            "/Users/einstalek/Downloads/without_glasses"]


if __name__ == '__main__':
    model = torch.jit.load(MODEL_PATH)
    _ = model.eval()

    for img_dir in IMG_DIRS:
        print("Going through {}".format(img_dir))
        for fp in os.listdir(img_dir):
            if '.jpg' not in fp:
                continue
            img = cv2.imread(os.path.join(img_dir, fp))[..., ::-1]
            img = cv2.resize(img, (128, 128)) / 255.
            img = (img - 0.5) / 0.5
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()[None]
            pred = model(img).detach().cpu().numpy().argmax()
            if pred == 1:
                print(os.path.join(img_dir, fp))
