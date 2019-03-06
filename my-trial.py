import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

def resize2SquareKeepingAspectRation(img, interpolation, size=64):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)


def main():
    with open('my_model_cat_dog', 'rb') as f:
        model = pickle.load(f)
    
    img1=cv2.imread('/home/jak/Desktop/trial/pushti.jpeg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.show()

    img1 = resize2SquareKeepingAspectRation(img1, cv2.INTER_AREA)
    array=[]
    array.insert(0,img1)
    x=np.asarray(array)
    if model.predict(x)==1:
        print("it's a dog")
    else:
        print("it's a cat")
        

if __name__== "__main__" :
    main()

