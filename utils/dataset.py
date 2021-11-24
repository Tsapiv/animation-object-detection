import torch
from torch.nn.functional import pad


def get_padding(image, shape):
    w, h = image.shape[-2:]
    h_padding = (shape[1] - w) / 2
    v_padding = (shape[0] - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class CentralPad:
    def __init__(self, shape, fill=0, padding_mode='constant'):
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.shape = shape
        self.padding_mode = padding_mode

    def __call__(self, img):
        return pad(img, list(get_padding(img, self.shape)), self.padding_mode, self.fill)

    def __repr__(self):
        return self.__class__.__name__ + '(fill={}, padding_mode={})'.format(self.fill, self.padding_mode)


def channel_expander(x):
    return x.repeat(3, 1, 1)
