import torch
import numpy as np
import copy
import nibabel as nib


def read_mask(path):
    mask = np.load(path, allow_pickle=True)
    arr = mask.get('arrays', mask.get('arr_0')).reshape(3, -1, 512, 512)
    liver = arr & 1
    tumor = arr >> 1
    return liver, tumor


def read_mask_kits19(path):
    seg = nib.load(path).get_fdata().astype('int8')
    tumor = seg >> 1
    seg[seg > 0] = 1
    return seg, tumor


def read_mask_brats20(path):
    seg = nib.load(path).get_fdata().astype('int8')
    return seg


def to_3class_onehot(img, bg=False):
    img1 = copy.deepcopy(img)
    img2 = copy.deepcopy(img)
    img3 = copy.deepcopy(img)
    img1[img1 != 1] = 0
    img2[img2 != 2] = 0
    img3[img3 != 3] = 0
    if bg:
        img0 = copy.deepcopy(img)
        img0 += 1
        img0[img0 != 1] = 0
        ret = np.stack((img0, img1, img2, img3), axis=0)
        ret[ret > 0] = 1
    else:
        ret = np.stack((img1, img2, img3), axis=0)
        ret[ret > 0] = 1
    return ret


def to_3channel_replica(img):
    """
    convert 1 channel gray image to 3 channel by replicate itself 3 times
    :param img:
    :return:
    """
    ret = np.stack((img, img, img), axis=0)
    return ret


def cut_384(img):
    """
    cut a 512*512 ct img to 385*384
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        ret = img[:, 50:434, 60:444]
    else:
        ret = img[50:434, 60:444]
    return ret


def pad_512(img):
    if len(img.shape) > 2:
        ret = np.pad(img, ((0, 0), (50, 78), (60, 68)), 'constant')
    else:
        ret = np.pad(img, ((50, 78), (60, 68)), 'constant')
    return ret


def clip_to_image(bbox, h, w):
    """
    keep bbox in the image
    """
    bbox[..., :2] = torch.clamp(bbox[..., :2], min=0)
    bbox[..., 2] = torch.clamp(bbox[..., 2], max=w-1)
    bbox[..., 3] = torch.clamp(bbox[..., 3], max=h-1)
    return bbox


def window_standardize(img, lower_bound, upper_bound):
    """
    clip the pixel values into [lower_bound, upper_bound], and standardize them
    """
    img = np.clip(img, lower_bound, upper_bound)
    # x=x*2-1: map x to [-1,1]
    img = 2 * (img - lower_bound) / (upper_bound - lower_bound) - 1
    return img


def draw_umich_gaussian(heat_map, center, h, w, k=1):
    """
    draw gaussian radius for 'center' in 'heat_map'
    :param heat_map: heat_map canvas
    :param center: gaussian center
    :param w: width
    :param h: height
    :param k:
    :return:
    """
    radius = _gaussian_radius((h, w))
    radius = max(0, int(radius))
    diameter = 2 * radius + 1
    gaussian = _gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heat_map.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heat_map = heat_map[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heat_map.shape) > 0:
        np.maximum(masked_heat_map, masked_gaussian * k, out=masked_heat_map)
    return heat_map


def _gaussian2d(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def _gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if b3 ** 2 - 4 * a3 * c3 < 0:
        r3 = min(r1, r2)
    else:
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def sample_normalize(vol):
    """
    :param vol: C(4) X W X H
    :return: vol after normalize in each channel
    """
    for i in range(vol.shape[0]):
        vol[i] = normalize(vol[i])
    return vol


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        tmp = 2 * (tmp- np.min(tmp)) / (np.max(tmp) - np.min(tmp)) -1
        # map x to [-1,1]
        return tmp
