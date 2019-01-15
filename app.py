from PIL import Image
from PIL import ImageOps
import os
from Model.model import Net
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse

def main() :
  args = process_args()
  model = load_model()
  invert = args.invert.lower() in ("true", "yes", "y")
  disp = args.disp.lower() in ("true", "yes", "y")
  img = load_image(args.img_path, invert, disp)
  predict(model, img)

def process_args():
  parser = argparse.ArgumentParser(description="hand written digit recognizer")
  package_dir = os.path.dirname(os.path.abspath(__file__))
  default_img_path = os.path.join(package_dir, 'assets', 'test4.png')
  parser.add_argument("--img-path", '-p', type=str, 
    help="specify image path",  metavar='image path',
    default=default_img_path)
  parser.add_argument("--invert", '-i', 
    default="False", metavar="invert color",
    help="to invert image color or not, default is \"False\", set \"True\" to invert the color.")
  parser.add_argument("--disp", '-d', 
    default="False", metavar="display image",
    help='to display image or not, default is "False", set "True" to display the image.') 
  return parser.parse_args()

def load_model():
  package_dir = os.path.dirname(os.path.abspath(__file__))
  model_path = os.path.join(package_dir, 'Model', 'model')
  model = Net()
  model.load_state_dict(torch.load(model_path))
  return model

def load_image(path, invert=False, disp=False):
  with Image.open(path).convert('L') as img:
    if invert:
      img = ImageOps.invert(img)
    if disp:
      plt.imshow(img)
      plt.show()
    transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((28, 28)),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0,), (1,))
    ])
    trans = transform(img)
    res = trans.unsqueeze(0)
    disp_img = torchvision.transforms.ToPILImage()(trans)
    if disp:
      plt.imshow(disp_img)
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