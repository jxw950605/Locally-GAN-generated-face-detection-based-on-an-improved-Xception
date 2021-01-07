import torch

def random_regular_mask(img):
    """Generates regular mask"""
    mask = torch.ones_like(img)
    size = img.size()
    limx = int(size[1] * 6 / 8)
    limy = int(size[2] * 6 / 8)
    #range_x,range_y is the scope for the mask in the image
    range_x = int(size[1] / 8)#size[1] / 8 is 1/8 of the side w of the image
    range_y = int(size[2] / 8)#size[1] / 8 is 1/8 of the side h of the image
    # x_fist y_fist is the starting coordinate of the upper left corner for the mask in the image
    x_fist = int(size[1] / 8)
    y_fist = int(size[2] / 8)
    x = random.randint(x_fist, int(limx))
    y = random.randint(y_fist, int(limy))
    mask[:, x:x + range_x, y:y + range_y] = 0

    return mask