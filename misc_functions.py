"""
Modified by wdf base on github.com/utkuozbulak
"""
import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def save_class_activation_images(org_img, activation_map, file_name,
                                 is_video=False):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
        is_video (bool): True means the input org_img is tensor, False means PIL image
    """
    if is_video:  # convert tensor to PIL
        org_img = transforms.ToPILImage()(org_img)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Save grayscale heatmap
    path_to_file = os.path.join('./results', file_name + '_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)

    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map,
                                                        'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('./results', file_name + '_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)

    # Save heatmap on image
    path_to_file = os.path.join('./results', file_name + '_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)

    # Save origin image
    path_to_file = os.path.join('./results', file_name + '_Cam_Ori_Image.png')
    save_image(org_img, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray(np.uint8((heatmap * 255)))
    no_trans_heatmap = Image.fromarray(np.uint8((no_trans_heatmap * 255)))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image,
                                             org_im.convert('RGBA'))
    # print(org_im.size, heatmap_on_image.size, heatmap.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr



def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)




from torchvision import transforms

def save_class_activation_images_as_gif(org_imgs,
                                        activation_maps,
                                        file_name,
                                        is_video=False):
    if is_video:
        org_imgs = org_imgs.permute(1, 0, 2, 3)
        org_imgs = [transforms.ToPILImage()(org_img) for org_img in org_imgs]

    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Save grayscale heatmap
    path_to_file = os.path.join('./results', file_name + '_Cam_Grayscale.gif')
    save_image_as_gif(activation_maps, path_to_file)

    # Grayscale activation map
    heatmap_on_images = []
    heatmaps = []
    origins = []
    for (org_img, activation_map) in zip(org_imgs, activation_maps):
        heatmap, heatmap_on_image = apply_colormap_on_image(org_img,
                                                            activation_map,
                                                            'hsv')
        heatmap_on_images.append(heatmap_on_image)
        heatmaps.append(heatmap)
        origins.append(org_img)

    # Save colored heatmap
    path_to_file = os.path.join('./results', file_name + '_Cam_Heatmap.gif')
    save_image_as_gif(heatmaps, path_to_file)

    # Save heatmap on image
    path_to_file = os.path.join('./results', file_name + '_Cam_On_Image.gif')
    save_image_as_gif(heatmap_on_images, path_to_file)

    # Save colored image
    path_to_file = os.path.join('./results', file_name + '_Cam_Ori_Images.gif')
    save_image_as_gif(org_imgs, path_to_file)


def save_image_as_gif(images, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    imgs = []
    for im in images:
        if isinstance(im, (np.ndarray, np.generic)):
            im = format_np_output(im)
            im = Image.fromarray(im)
        imgs.append(im)
    # im.save(path)
    imgs[0].save(path, save_all=True, append_images=imgs, duration=3)
