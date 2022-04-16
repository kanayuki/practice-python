import os
import sys

from PIL import Image

p1 = r"Y:\IMAGE\BEAUTYLEG\[Be]2021.05.19 No.2079 ChiChi[47P562M]\0016.jpg"
p2 = r"Y:\IMAGE\BEAUTYLEG\[Be]2021.05.19 No.2079 ChiChi[47P562M]\0027.jpg"

# %%
img = Image.open(p1)
img.show('ChiChi')

# %%
print(img.format, img.size, img.mode)

# %% 转换图片成JPG

for infile in sys.argv[1:]:
    f, e = os.path.splitext(infile)
    outfile = f + ".jpg"
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                im.save(outfile)
        except OSError:
            print("cannot convert", infile)

# %% 创建缩略图
size = (128, 128)

for infile in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0] + ".thumbnail"
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                im.thumbnail(size)
                im.save(outfile, "JPEG")
        except OSError:
            print("cannot create thumbnail for", infile)


# %%
def roll(im, delta):
    """Roll an image sideways."""
    xsize, ysize = im.size

    delta = delta % xsize
    if delta == 0:
        return im

    part1 = im.crop((0, 0, delta, ysize))
    part2 = im.crop((delta, 0, xsize, ysize))
    im.paste(part1, (xsize - delta, 0, xsize, ysize))
    im.paste(part2, (0, 0, xsize - delta, ysize))

    return im


roll(img, 1000)
img.show()


# %%
def merge(im1, im2):
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGBA", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im


img2 = Image.open(p2)
im = merge(img, img2)
im.show()
# %%
out = im.resize((128, 128))
out = out.rotate(45)  # degrees counter-clockwise
out.show()
