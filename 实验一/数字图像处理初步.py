from PIL import Image

if __name__ == "__main__":
    img = Image.open('logo.jpg')
    assert isinstance(img, Image.Image)
    w, h = img.size
    print(img.format, img.size, img.mode)
    w = w / 2
    h = h / 2
    img2 = img.resize((int(w), int(h)))
    print(img2.format, img2.size, img2.mode)
    img3bw = img2.convert('1')
    print(img3bw.format, img3bw.size, img3bw.mode)
    img2.save('logo2.png')
    img3bw.save('logo2bw.png')