from PIL import Image
from PIL import ImageChops

if __name__ == "__main__":
    img1 = Image.open('4.2.05.tiff')
    assert isinstance(img1, Image.Image)
    img2 = Image.open('4.2.06.tiff')
    assert isinstance(img2, Image.Image)
    print(img1.size, img2.size)
    img_add = ImageChops.add(img1, img2)
    assert isinstance(img_add, Image.Image)
    img_sub = ImageChops.subtract(img1, img2)
    assert isinstance(img_sub, Image.Image)
    img_multiply = ImageChops.multiply(img1, img2)
    assert isinstance(img_multiply, Image.Image)
    # img_add.show()
    # img_sub.show()
    # img_multiply.show()
    img_divide = Image.new('RGB', (512, 512), 0)
    for x in range(512):
        for y in range(512):
            r1, g1, b1 = img1.getpixel((x, y))
            r2, g2, b2 = img2.getpixel((x, y))
            if r2:
                r3 = int(r1 / r2)
            else:
                r3 = r1
            if g2:
                g3 = int(g1 / g2)
            else:
                g3 = g2
            if b2:
                b3 = int(b1 / b2)
            else:
                b3 = b2
            img_divide.putpixel((x, y), (r3, g3, b3))

    img_add.save('add.jpg')
    img_sub.save('sub.jpg')
    img_multiply.save('multiply.jpg')
    img_divide.save('divide.jpg')
