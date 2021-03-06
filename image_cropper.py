from PIL import Image
import sys
import numpy as np
import imageio


"""
Crop to the desired input size for our model (28,28)
If the image has more than one channel (e.g. 4 in the case of screenshotted pngs), then convert to grayscale
"""


def test(filename, to_filename):

    ifile = imageio.imread(filename)
    ofile = imageio.imread(to_filename)

    print("Input size {} Output size {}".format(ifile.shape, ofile.shape))

def eliminate_whitespace(to_filename, image_buffer=2):
    '''
    finds outermost nonwhite
    pixels and crops out white pixels
    outside of the bounds
    returns cropped Pillow Image

    **** note: initial image must have enough whitespace around
    it to create the white margins. Else the pixel values which
    strech beyond the bounds of the original image are automatically
    filled in as black
    '''
    img = imageio.imread(to_filename) #read in as np array
    nonwhite_pixels = np.argwhere(img < 255) # not white
    num_cols = nonwhite_pixels.shape[0]
    
    upper = nonwhite_pixels[0,0]
    lower = nonwhite_pixels[num_cols-1,0]
    left = np.min(nonwhite_pixels[np.arange(num_cols),1])
    right = np.max(nonwhite_pixels[np.arange(num_cols),1])
    height = lower - upper
    width = right - left

    assert (height > 0)
    assert (width > 0)

    vertical_dim_bigger = height > width
    horizontal_dim_bigger = width > height
    if vertical_dim_bigger:
        margin = (height - width) // 2
        left -= margin
        right += margin
    elif horizontal_dim_bigger:
        margin = (width - height) // 2
        upper -= margin
        lower += margin
    left -= image_buffer
    right += image_buffer
    upper -= image_buffer
    lower += image_buffer
    box = (left, upper, right, lower)
    img = Image.open(to_filename)
    img = img.crop(box)
    return img

def main():

    print(sys.argv)
    filename = sys.argv[1]
    to_filename = sys.argv[2]
    image_buffer = None
    if len(sys.argv) == 4:
        image_buffer = sys.argv[3]
    img = Image.open(filename).convert('L')
    img.save(to_filename)
    if image_buffer is not None:
        img = eliminate_whitespace(to_filename, int(image_buffer))
    else:
        img = eliminate_whitespace(to_filename)
    img.save(to_filename)
    img = img.resize((28,28))
    img.save(to_filename)


    # Read back in as a numpy array
    img = imageio.imread(to_filename)
    img = 255 - img
    imageio.imsave(to_filename, img)

    test(filename, to_filename)


if __name__=="__main__":
    main()