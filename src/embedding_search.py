print('[INFO]: Importing packages')
# geneal tools
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from tqdm import tqdm

# tensorflow
from tensorflow.keras.preprocessing.image import (load_img, img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, preprocess_input)

# scikit-learn
from sklearn.neighbors import NearestNeighbors

# matplotlib
import matplotlib.pyplot as plt

def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)
    """
    # Define input image shape
    input_shape = (224, 224, 3)
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img, verbose=0)
    # flatten
    flattened_features = features.flatten()
    # normalise features
    normalized_features = flattened_features / norm(features)
    
    return normalized_features

def main():

    # load VGG16
    print('[INFO]: Loading VGG16')
    model = VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(224, 224, 3))
    
    # relative path to flower images
    root_dir = os.path.join('..', 'data', 'flowers')
    filenames = [os.path.join(root_dir, name) for name in sorted(os.listdir(root_dir))]

    # sort out none-files
    filenames = [filename for filename in filenames if os.path.isfile(filename)]

    # if embeddings have not been made previously
    if not os.path.exists( os.path.join('..', 'data', 'flower_embeds.pkl') ):
        # extract features from all images
        print('[INFO]: Computing VGG16 embeddings for all images')
        feature_list = [extract_features(filename, model) for filename in tqdm(filenames)]

        # save embeddings
        outpath = os.path.join('..', 'data', 'flower_embeds.pkl')
        with open(outpath, 'wb') as file:
            pd.to_pickle(feature_list, file)

    # load saved embeddings
    inpath = os.path.join('..', 'data', 'flower_embeds.pkl')
    feature_list = pd.read_pickle(inpath)
    
    # initialise KNN algorithm
    neighbors = NearestNeighbors(n_neighbors=6, 
                                algorithm='brute',
                                metric='cosine').fit(feature_list)

    # find closest neighbors
    print('[INFO]: Identifying closest images')
    distances, indices = neighbors.kneighbors([feature_list[0]])

    # unpack distances and indices
    dists = [round(dist, 3) for dist in distances[0]]
    idxs = [idx for idx in indices[0]]


    # find corresponding image files
    image_files = [filenames[idx][-14:] for idx in idxs]

    # create overview dataframe
    dict = {'image_filename': image_files,
            'distance': dists}
    
    df = pd.DataFrame(dict)

    df.to_csv(os.path.join('..', 'out', 'top5embedsIMG0001.csv'), index=False)

    # make figure of chosen flower along with five closets images
    ## prepare list of images
    images = [plt.imread(filenames[idx]) for idx in idxs]

    ## prepare a list of image titles
    titles = ['Chosen flower']
    for dist in dists[1:]:
        titles.append(f'Distance: {str(dist)}')

    plt.style.use('seaborn-white')

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9,6))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(f"{titles[i]}")

    plt.tight_layout()
    plt.savefig(os.path.join("..","out","top5embedsIMG0001.png"))

if __name__ == '__main__':
    main()