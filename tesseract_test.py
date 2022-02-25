from PIL import Image, ImageDraw
import pytesseract
import torch
from data import CharacterDataset
from train import CharacterClassifier
from pdf2image import convert_from_path
from torch.utils.data import DataLoader
import os
import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", type=str, required=True)
    args = parser.parse_args()

    # Convert pdf to images
    os.makedirs("pdf_images", exist_ok=True)
    paths = convert_from_path(args.pdf_path, output_folder="pdf_images", output_file="pdf_page-", paths_only=True, use_pdftocairo=True)

    device = torch.device("cuda:0")
    model = torch.load("model.pt")

    for page_path in paths:
        # Get character bboxes from tesseract. Use legacy model since it has better character bboxes
        data = pytesseract.image_to_boxes(page_path, output_type="dict", config=r"--oem 0")

        image = Image.open(page_path)
        image.load()
        w, h = image.size
        draw = ImageDraw.Draw(image)

        # Crop characters
        char_crops = []
        for left, top, right, bottom in tqdm.tqdm(zip(data["left"], data["top"], data["right"], data["bottom"]), total=len(data["left"])):
            # draw.rectangle((left, h - top, right, h - bottom), outline=(255, 0, 0))
            char_crop = image.crop((left, h - top, right, h - bottom))
            img_size = 32
            # assert char_crop.width <= img_size and char_crop.height <= img_size
            # Letterbox image so that its size is img_size x img_size, same as during training
            char_img = CharacterDataset.letterbox_image(char_crop, img_size, 255)
            char_crops.append(CharacterDataset.pil_to_torch(char_img))

        data_loader = DataLoader(char_crops, batch_size=32, shuffle=False)

        # Add formatting info to tesseract data object
        data["bold"] = [False] * len(data["char"])
        data["italic"] = [False] * len(data["char"])

        i = 0
        for batch in data_loader:
            batch = batch.to(device)
            with torch.no_grad():
                logits = model(batch)
                predictions = logits >= 0
                bold, italic = predictions[:, 0], predictions[:, 1]

            # Draw predictions on document: bold+italic is red, bold is green, italic is blue
            for j in range(len(batch)):
                if bold[j] or italic[j]:
                    bbox = data["left"][i+j], h - data["top"][i+j], data["right"][i+j], h - data["bottom"][i+j]
                    if bold[j] and italic[j]:
                        color = (255, 0, 0)
                    elif bold[j]:
                        color = (0, 255, 0)
                    else:  # italic
                        color = (0, 0, 255)
                    draw.rectangle(bbox, outline=color)
                    data["bold"][i + j] = bool(bold[j])
                    data["italic"][i + j] = bool(italic[j])
            i += len(batch)
        image.show()

        # `data` now has formatting information and could be used elsewhere


if __name__ == "__main__":
    main()
