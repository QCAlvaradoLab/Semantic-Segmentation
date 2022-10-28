from torchvision.transforms import Lambda
from PIL import Image, ImageFilter
import numpy as np

class EnhancedCompose(object):
    """Composes several transforms together, support separate transformations for multiple input.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')                
        return img

class Merge(object):
    """Merge a group of images
    """
    def __init__(self, axis=-1):
        self.axis = axis
    def __call__(self, images):
        if isinstance(images, collections.Sequence) or isinstance(images, np.ndarray):
            assert all([isinstance(i, np.ndarray) for i in images]), 'only numpy array is supported'
            shapes = [list(i.shape) for i in images]
            for s in shapes:
                s[self.axis] = None
            assert all([s==shapes[0] for s in shapes]), 'shapes must be the same except the merge axis'
            return np.concatenate(images, axis=self.axis)
        else:
            raise Exception("obj is not a sequence (list, tuple, etc)")

class Split(object):
    """Split images into individual images
    """
    def __init__(self, *slices, **kwargs):
        assert isinstance(slices, collections.Sequence)
        slices_ = []
        for s in slices:
            if isinstance(s, collections.Sequence):
                slices_.append(slice(*s))
            else:
                slices_.append(s)
        assert all([isinstance(s, slice) for s in slices_]), 'slices must be consist of slice instances'
        self.slices = slices_
        self.axis = kwargs.get('axis', -1)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            ret = []
            for s in self.slices:
                sl = [slice(None)]*image.ndim
                sl[self.axis] = s
                ret.append(image[sl])
            return ret
        else:
            raise Exception("obj is not an numpy array")

def GaussianBlur(std):
# std: 1 to 2 or 3
    return Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(std)))

def PoissonBlur(peak):
# PEAK: 0 to 1 defining intensity of noise
    def poisson_blur_f(image, peak):
        noisy = np.random.poisson(np.array(image) / 255.0 * peak) / peak * 255  # noisy image
        return Image.fromarray(noisy.astype(np.uint8))
    return Lambda(lambda img: poisson_blur_f(img, peak))

def GaussianAdditiveNoise(std, percent):
# percent: 0 to 1
    
    def gaussian_noise_f(image, std, percent):
        image = np.array(image)
        noise = np.random.normal(loc=0, scale=std, size=image.shape)
        return Image.fromarray(np.array(image + percent*noise, dtype=np.uint8))
    
    return Lambda(lambda img: gaussian_noise_f(img, std, percent))

def PoissonAdditiveNoise(peak, percent):
# percent: 0 to 1 (1)
# PEAK: 0.1 to 0.9
    poiss_tr = PoissonBlur(peak)

    def poisson_noise_f(image, percent):
        noise = np.array(poiss_tr(image))
        image = image + percent*noise
        return Image.fromarray(image.astype(np.uint8))
    return Lambda(lambda img: poisson_noise_f(img, percent))

def SaltandPepperAdditiveNoise(percent):
# percent: 0 to 1
    
    def salt_and_pepper_f(image, amount, s_vs_p=0.5):
        image = np.array(image)
        row,col,ch = image.shape
        out = image.copy()
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
        out[tuple(coords)] = 0 

        return Image.fromarray(out)

    return Lambda(lambda img: salt_and_pepper_f(img, percent))

if __name__=='__main__':
    
    tr = GaussianBlur(0.01)
    img = Image.open('/home/hans/sample9.jpg')
    
    img.save('sample.jpg')
    trimg = tr(img)
    trimg.save('sample_noise.jpg')
