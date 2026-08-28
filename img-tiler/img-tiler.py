from PIL import Image
from itertools import product
import os

# Source - https://stackoverflow.com/a/65698752
# Posted by Ivan, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-19, License - CC BY-SA 4.0

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

def main():

    tile('DJI_20260512114933_0001_V - Copy.JPG', 'experiments', 'experiments', 128)

if __name__ == '__main__':
    main()