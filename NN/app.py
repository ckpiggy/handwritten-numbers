from PIL import Image
from PIL import ImageOps
import os
from model import Net
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse

def main() :
  args = process_args()
  model = load_model()
  invert = False
  if args.invert.lower() in ("true", "yes", "y"):
    invert = True
  img = load_image(args.img_path, invert)
  predict(model, img)

def process_args():
  parser = argparse.ArgumentParser(description="hand written digit recognizer")
  package_dir = os.path.dirname(os.path.abspath(__file__))
  default_img_path = os.path.join(package_dir, 'assets', 'test5.png')
  parser.add_argument("--img-path", type=str, help="image path", default=default_img_path)
  parser.add_argument("--invert", default="False", metavar="I",
    help="to invert image color or not, default is \"False\", set \"True\" to invert the color.")
  return parser.parse_args()

def load_model():
  package_dir = os.path.dirname(os.path.abspath(__file__))
  model_path = os.path.join(package_dir, 'model')
  model = Net()
  model.load_state_dict(torch.load(model_path))
  return model

def load_image(path, invert=False):
  with Image.open(path).convert('L') as img:
    if invert:
      img = ImageOps.invert(img)
    plt.imshow(img)
    plt.show()
    transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((28, 28)),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0,), (1,))
    ])
    trans = transform(img)
    res = trans.unsqueeze(0)
    disp = torchvision.transforms.ToPILImage()(trans)
    plt.imshow(disp)
    plt.show()
    return res

def predict(model, img):
  model.eval()
  output = model(img)
  max = output.max(1)
  pred = max[1][0].item()
  conf = max[0][0].item()
  if conf < 3:
    print("hmm... I don't think that is a digit")
  else:
    print("this could be {0}".format(pred))
  

if __name__ == "__main__":
  main()    