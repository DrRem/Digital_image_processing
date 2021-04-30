from PIL import Image

if __name__ == "__main__":
    im  = Image.open('7.2.01.tiff')
    assert isinstance(im, Image.Image)
    im.