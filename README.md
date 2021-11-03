# ComicGen with rationals


## Install
We advise to install in a virtual environment:
`python3 -m venv env`
`source env/bin/activate`

To install
* `pip3 install -U pip wheel`
* `pip3 install -r requirements.txt`
* Install [the rational activation function library](https://github.com/ml-research/rational_activations).

## How to use
To cartoonize your images, please
* place your images in the `images` folder
* run `python3 cartoonize.py`
* results are available in `cartoonized_images `



## To retrain the models
### Pretrain
- Place your training data in corresponding folder in /dataset with the following structure
        dataset
        |_ cartoon_face
	   |_ ...
	   |_ *.jpg
        |_ cartoon_scenery
	   |_ ...
	   |_ *.jpg
        |_ photo_face
	   |_ ...
	   |_ *.jpg
        |_ photo_scenery
	   |_ ...
	   |_ *.jpg
- Run pretrain.py, results will be saved in /pretrain/images folder

### Train
- Firstly, please run pretrain.py to pretrain the model
- Then run train.py, results will be saved in /train_cartoon/images folder
- Pretrained VGG_19 model can be found at following url:
https://drive.google.com/file/d/1j0jDENjdwxCDb36meP6-u5xDBzmKBOjJ/view?usp=sharing
