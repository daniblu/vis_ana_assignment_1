print('[INFO]: Importing packages')
import os
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    # load image
    flowers_path = os.path.join("..", "data", "flowers")
    chosen_flower = cv2.imread(os.path.join(flowers_path, "image_0001.jpg"))

    # draw histogram and normalize range
    chosen_flower_hist = cv2.calcHist([chosen_flower], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    chosen_flower_hist = cv2.normalize(chosen_flower_hist, chosen_flower_hist, 0, 1.0, cv2.NORM_MINMAX)

    #list all files (and files only) in flower directory
    flowers = []
    for entry in os.listdir(flowers_path):
        if os.path.isfile(os.path.join(flowers_path, entry)):
            flowers.append(entry)

    # loop through flowers and compare
    print("[INFO]: Calculating distances to each image")
    distances = []
    for flower in tqdm(flowers):
        # load image
        flower_to_be_compared = cv2.imread(os.path.join(flowers_path, flower))
        # draw histogram and normalize
        flower_hist = cv2.calcHist([flower_to_be_compared], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
        flower_hist = cv2.normalize(flower_hist, flower_hist, 0, 1.0, cv2.NORM_MINMAX)
        # compute distance to chosen flower 
        distance = round(cv2.compareHist(chosen_flower_hist, flower_hist, cv2.HISTCMP_CHISQR), 2)
        # append filename and distance to a list
        distances.append((flower, distance))

    # change to data frame
    df = pd.DataFrame(distances, columns=["image_filename", "distance"])

    # sort distances in ascending order
    df = df.sort_values(by = "distance")

    # isolate the five closest images (first list entry is the chosen flower itself)
    top = df.iloc[:6,]

    # write csv
    top.to_csv(os.path.join("..", "out", "top5histosIMG0001.csv"), index=False)

    # make figure of chosen flower along with five closets images
    ## prepare list of images
    images = []
    for flower in df.iloc[:6,0]:
        rel_path = os.path.join("..", "data", "flowers", flower)
        image = plt.imread(rel_path)
        images.append(image)
    
    ## prepare a list of image titles
    titles = ['Chosen flower', 
              f'Distance: {df.iloc[1,1]}', 
              f'Distance: {df.iloc[2,1]}', 
              f'Distance: {df.iloc[3,1]}', 
              f'Distance: {df.iloc[4,1]}', 
              f'Distance: {df.iloc[5,1]}']

    plt.style.use('seaborn-white')

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9,6))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(f"{titles[i]}")

    plt.tight_layout()
    plt.savefig(os.path.join("..","out","top5histosIMG0001.png"))

if __name__ == "__main__":
    main()