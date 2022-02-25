from PIL import Image, ImageDraw, ImageFont
import string
import numpy as np
import os
import torch
from torch.utils.data import Dataset

FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/"
FONTS = [
    "Arial",
    "Times_New_Roman",
    "Courier_New",
]

VOCABULARY = list(string.ascii_letters + string.digits + ".,\"-")


class CharacterDataset(Dataset):
    # Seed is used to make sure we always generate the same dataset
    def __init__(self, size, seed):
        super().__init__()
        self.size = size
        self.seed = seed
        self.cache = {}

    def __len__(self):
        return self.size

    @staticmethod
    def pil_to_torch(image):
        image = np.array(image) / 255.0
        image = (image - 0.5) / 0.5
        return torch.from_numpy(image).type(torch.float32).unsqueeze(0)

    @staticmethod
    def letterbox_image(image, img_size, background_color):
        w, h = image.width, image.height

        # Rescale image to have larger side img_size
        if w > h:
            new_w, new_h = img_size, h * img_size // w
        else:
            new_w, new_h = w * img_size // h, img_size
        image = image.resize((new_w, new_h))

        # Paste back into empty image
        new_image = Image.new("L", (img_size, img_size), background_color)
        new_image.paste(image, ((img_size - new_w) // 2, (img_size - new_h) // 2))

        return new_image

    def __getitem__(self, index):
        # Cache computation
        if index in self.cache:
            return self.cache[index]

        # Save rng state so we can restore it later
        rng_state = np.random.get_state()
        # Use seed + index as seed so that we always generate the same instances no matter in what order
        # the dataset is accessed
        np.random.seed(self.seed + index)

        # Random background color
        background_color = 255  # np.random.randint(240, 255 + 1)

        # Random font
        font_type = np.random.choice(FONTS)
        font_size = np.random.randint(15, 25 + 1)
        font_fill = 0  # np.random.randint(0, 15 + 1)
        if np.random.random() < 0.25:
            bold = np.random.randint(0, 2)
            italic = np.random.randint(0, 2)
            if bold:
                font_type += "_Bold"
            if italic:
                font_type += "_Italic"
        else:
            bold = italic = 0
        font_path = os.path.join(FONT_PATH, f"{font_type}.ttf")
        font = ImageFont.truetype(font_path, font_size)

        # Random character
        char = np.random.choice(VOCABULARY)

        # First draw character onto background and get character crop
        img_size = 32

        image = Image.new("L", (img_size, img_size), background_color)
        draw = ImageDraw.Draw(image)
        box = list(draw.textbbox((0, 0), char, font=font))
        w, h = box[2] - box[0], box[3] - box[1]

        assert w <= img_size and h <= img_size
        draw.text(((img_size - w) // 2, (img_size - h) // 2), char, fill=font_fill, font=font)

        # Shift box to center
        box[0] += (img_size - w) // 2
        box[2] += (img_size - w) // 2
        box[1] += (img_size - h) // 2
        box[3] += (img_size - h) // 2

        # Randomly make box coordinates by up to 10% since during inference we usually get imperfect/tighter
        # bounding boxes
        max_x_noise = int(w * 0.1)
        max_y_noise = int(h * 0.1)

        if max_x_noise > 0:
            box[0] = np.clip(box[0] + np.random.randint(max_x_noise // 2, max_x_noise+1), 0, img_size)
            box[2] = np.clip(box[2] + np.random.randint(max_x_noise // 2, max_x_noise+1), 0, img_size)
        if max_y_noise > 0:
            box[1] = np.clip(box[1] + np.random.randint(-max_y_noise, max_y_noise // 2 + 1), 0, img_size)
            box[3] = np.clip(box[3] + np.random.randint(-max_y_noise, max_y_noise // 2 + 1), 0, img_size)

        # Letterbox image to make sure the scale is fixed and the image is square
        char_crop = image.crop(tuple(box))

        image = self.letterbox_image(char_crop, img_size, background_color)

        self.cache[index] = self.pil_to_torch(image), torch.LongTensor([bold, italic])

        np.random.set_state(rng_state)
        return self.cache[index]
