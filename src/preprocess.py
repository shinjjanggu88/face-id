from PIL import Image
image = Image.open("소희/소앞.jpg")
image = image.resize((128, 128 ))
image.save("소희/소앞_128.jpg")