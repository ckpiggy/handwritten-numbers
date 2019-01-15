## A Pytorch CNN demo app.
This app train a CNN to recognize handwriiten digits use MNIST dataset.

### To train your model

```
> python3 ./Model/train.py 
```
I got a 98% accuracy model with 3 epochs momentum 0.9.
```
> python3 ./Model/train.py --epochs=3 --momentum-0.9
```

### To recognize the digits

```
> python3 app.py -h
usage: app.py [-h] [--img-path image path] [--invert invert color]
              [--disp display image]

hand written digit recognizer

optional arguments:
  -h, --help            show this help message and exit
  --img-path image path, -p image path
                        specify image path
  --invert invert color, -i invert color
                        to invert image color or not, default is "False", set
                        "True" to invert the color.
  --disp display image, -d display image
                        to display image or not, default is "False", set
                        "True" to display the image.
```
For example:
```
> python3 app.py -p=./assets/test9.png -d=True -i=True
this could be 9
```
It will load the image in assets folder, display the image during process and invert the color before recognize it.

```
> python3 app.py -p=./assets/testcat.png -d=True
hmm... I don't think that is a digit
```
It will complain the you feed something strange 😎

**Howerver the model works poor when the background is lighter than foreground!**

The data we used to train the model got black background and white foreground. So you may need to invert the color before feed the them into the model 😛. 
