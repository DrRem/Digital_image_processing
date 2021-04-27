from PIL import Image

if __name__ == "__main__":
    img1 = Image.open('4.2.05.tiff')
    assert isinstance(img1, Image.Image)
    img2 = Image.open('4.2.06.tiff')
    assert isinstance(img2, Image.Image)
    print(img1.size, img2.size)