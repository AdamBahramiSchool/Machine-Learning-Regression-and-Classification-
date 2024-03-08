import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from skimage.color import rgb2lab as convert_rgb2lab
from sklearn.pipeline import make_pipeline
# command line argument: python3 colour_bayes.py colour-data.csv
# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 113, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 186, 186),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=70, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)

def rgb2lab(rgb_data):
    lab_data = rgb2lab_wrapper(rgb_data)
    return lab_data

def rgb2lab_wrapper(rgb_data):
    lab_dimension_data = convert_rgb2lab(rgb_data.reshape(-1, 1, 3)).reshape(-1, 3)
    return lab_dimension_data


def main(infile):
    data = pd.read_csv(infile)
    RGB_data=np.array(data[['R','G','B']])
    RGB_data=RGB_data/255
    Colour_data=np.array(data['Label'])
    X=RGB_data
    Y=Colour_data
    x_training_data,x_valid_data, y_training_data, y_valid_data=train_test_split(X,Y)
    # X = data # array with shape (n, 3). Divide by 255 so components are all 0-1.
    # y = data # array with shape (n,) of colour words.
    model_rgb=GaussianNB()
    model_rgb.fit(x_training_data,y_training_data)    
    print('model_rgb score: ', model_rgb.score(x_valid_data,y_valid_data))
    model_lab=make_pipeline(
        FunctionTransformer(rgb2lab_wrapper),
        GaussianNB()
    )
    model_lab.fit(x_training_data,y_training_data)
    print('model_lab score: ', model_lab.score(x_valid_data, y_valid_data))
    

    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')


if __name__ == '__main__':
    main(sys.argv[1])
