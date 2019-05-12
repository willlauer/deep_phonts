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

def eliminate_whitespace(to_filename, image_buffer=3):
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
    import pdb
    img = imageio.imread(to_filename) #read in as np array
    nonwhite_pixels = np.argwhere(img < 255) # not white
    num_cols = nonwhite_pixels.shape[0]
    upper = nonwhite_pixels[0,0]
    lower = nonwhite_pixels[num_cols-1,0]
    left = np.min(nonwhite_pixels[np.arange(num_cols),1])
    right = np.max(nonwhite_pixels[np.arange(num_cols),1])
    # pdb.set_trace()
    height = lower - upper
    width = right - left

    vertical_dim_bigger = height > width
    if vertical_dim_bigger:
        margin = (height - width) / 2
        left -= margin
        right += margin
    else:
        margin = (width - height) / 2
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

    ### TODO: crop excess whitespace here
    ### TODO: before here

    img = Image.open(filename).convert('L')
    img.save(to_filename)
    img = eliminate_whitespace(to_filename)
    img.save(to_filename)
    # input()
    img = img.resize((28,28))
    img.save(to_filename)


    # Read back in as a numpy array
    img = imageio.imread(to_filename)

    # Check if we need conversion to greyscale
    # if len(img.shape) > 2:

    #     img = img[:, :, :3] # pull rgb channels

    #     if np.max(img, 2).any() > 1: # we're in the 0-255 scale

    #         img = img / 255

    #     c_linear_r = 0.2126 * img[:,:,0]
    #     c_linear_g = 0.7152 * img[:,:,1]
    #     c_linear_b = 0.0722 * img[:,:,2]

    #     c_linear = c_linear_r + c_linear_g + c_linear_b

    #     print(c_linear.shape)

    #     img = 12.92 * c_linear * np.where(c_linear <= 0.0031308, 1, 0) \
    #           + (1.055 * (c_linear ** (1/2.4)) - 0.055) * np.where(c_linear > 0.0031308, 1, 0)


    #     imageio.imsave(to_filename, img)

    test(filename, to_filename)


if __name__=="__main__":
    main()