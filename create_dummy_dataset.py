import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs('dataset', exist_ok=True)

for i in range(1, 6):
    img = Image.new('RGB', (400, 400), color=(139, 0, 0)) # dark red background
    d = ImageDraw.Draw(img)
    # Add a yellowish circle for the optic disc
    d.ellipse([(250, 150), (330, 250)], fill=(255, 204, 102))
    # Add some text
    d.text((10,10), f"Fundus Sample {i}", fill=(255,255,255))
    img.save(f'dataset/sample_{i}.jpg')
print("Dummy dataset created.")
