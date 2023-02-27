import numpy as np
import cv2
import PIL
import pandas as pd
import stringprep
from PIL import Image
import pytesseract
from glob import glob
from tqdm import tqdm
import os
from pdf2image import convert_from_path
from pathlib import Path
import re
import string
import tempfile
import matplotlib.pyplot as plt
import spacy
import easyocr
from spacy import displacy
import calendar
import datetime

IMAGE_SIZE = 1800
BINARY_THRESHOLD = 180

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\bbarenthien\AppData\Local\Tesseract-OCR\tesseract.exe'


class GroupGen:
    def __init__(self):
        self.id = 0
        self.text = ""

    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id


grp_gen = GroupGen()

# Load NER model
model_ner = spacy.load("./output/model-best")

# img_path = glob("./temp/src-58dc2a82-f221-4776-aa9c-da43507119b2.jpg")
reader = easyocr.Reader(['de'], gpu=False)
# imgPaths = glob("./timesheets/jpegs/*.jpg")

# img_path = img_path[0]
# _, filename = os.path.split(img_path)


# def convert_pdf():
#     # assign directory
#     path = '/content/drive/MyDrive/AI_Data/test'
#     _, filename = os.path.split(img_path)
#     for filename in os.listdir(path):
#         if filename.endswith(".pdf"):
#             images = convert_from_path(path + '/' + filename)
#             stem = Path(filename).stem
#             for i in range(len(images)):
#                 filename = os.path.join(path, stem)
#                 images[i].save(filename + '.tif', 'TIFF')
#             continue
#         else:
#             continue

def convert_pdf(image_path):
    # assign directory
    print(image_path)
    path, file = os.path.split(image_path)
    for file in os.listdir(path):
        print(path)
        print(file)
        if file.endswith(".pdf"):
            images = convert_from_path(path + '/' + file)
            stem = Path(file).stem
            # print(stem)
            for i in range(len(images)):
                file = os.path.join(path + '\\jpegs\\', stem)
                images[i].save(file + '.jpg', 'JPEG')
                # images[i].save(file + '.tif', 'TIFF')
                image_file = os.path.join(path + '\\jpegs\\', stem+".jpg")
                # return image_file
            # continue
        else:
            continue


def new_convert_pdfs(folder):
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            images = convert_from_path(folder + '/' + file)
            stem = Path(file).stem
            for i in range(len(images)):
                # file = os.path.join(folder + '\\jpegs\\', stem)
                file = os.path.join(folder, stem)
                images[i].save(file + '.jpg', 'JPEG')
        else:
            continue


def new_process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    img_path = convert_pdf(file_path)
    temp_filename = set_image_dpi(img_path)
    im_new, original_image = enhance_image(temp_filename)
    return im_new, original_image


def process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    img_path = convert_pdf(file_path)
    temp_filename = new_set_image_dpi(img_path)
    im_new, original_image = remove_noise_and_smooth(temp_filename)
    return im_new, original_image


def train_process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    #img_path = convert_pdf(file_path)
    temp_filename = train_set_image_dpi(file_path)
    path, file = os.path.split(file_path)
    print(path)
    im_new, original_image = train_remove_noise_and_smooth(temp_filename, path)
    return im_new, original_image


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.Resampling.LANCZOS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename


def new_set_image_dpi(file_path):
    image_resize = Image.open(file_path)
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix='.tif')
    temp_filename = temp_file.name
    image_resize.save(temp_filename, dpi=(300, 300))
    return temp_filename


def train_set_image_dpi(file_path):
    print(file_path)
    print(type(file_path))
    image_resize = Image.open(file_path, "r")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    temp_filename = temp_file.name
    image_resize.save(temp_filename, dpi=(300, 300))
    return temp_filename


def image_smoothening(img):
    # ret1, th1 = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # blur = cv2.GaussianBlur(th2, (1, 1), 0)
    # ret3, th3 = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    img_cv2 = cv2.imread(file_name)

    # Test the following line
    c_img = apply_brightness_contrast(img, 0, 20)

    # detail_img = cv2.detailEnhance(c_img, sigma_s=10, sigma_r=0.1)
    filtered = cv2.adaptiveThreshold(c_img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image, img_cv2


def train_remove_noise_and_smooth(file_name, path):
    print(path)
    img = cv2.imread(file_name, 0)
    img_cv2 = cv2.imread(file_name)

    # Test the following line
    c_img = apply_brightness_contrast(img, 0, 20)

    # detail_img = cv2.detailEnhance(c_img, sigma_s=10, sigma_r=0.1)
    filtered = cv2.adaptiveThreshold(c_img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    path_tif, file = os.path.split(file_name)
    tif_path = os.path.join(path, file)
    cv2.imwrite(tif_path, or_image)

    cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original image", img_cv2)
    cv2.namedWindow("processed image", cv2.WINDOW_NORMAL)
    cv2.imshow("processed image", or_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return or_image, img_cv2


def enhance_image(image_file):
    image = cv2.imread(image_file)
    img_cv2 = image.copy()
    # cv2.imshow("Timesheet", img_cv2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    bright_contrast_img = apply_brightness_contrast(image, 0, 20)
    detail_img = cv2.detailEnhance(bright_contrast_img, sigma_s=10, sigma_r=0.1)
    # converting image into gray scale image
    gray_image = cv2.cvtColor(detail_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # converting it to binary image by Thresholding
    # this step is require if you have colored image because if you skip this part
    # then tesseract won't able to detect text correctly and this will give incorrect result
    threshold_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    img_cv2 = cv2.resize(img_cv2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    threshold_img = cv2.resize(threshold_img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    return threshold_img, img_cv2


def extract_text(image):
    # Text from image extraction
    data_cv = pytesseract.image_to_data(image, lang="deu")
    return data_cv


def cleantext(txt):
    whitespace = string.whitespace
    punctuation = '|!"#$%()*+-/:;<=>?@[\\]^_`{|}~'
    tableWhitespace = str.maketrans('', '', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    # text = text.lower()
    # removewhitespace = text.translate(tableWhitespace)
    removepunctuation = text.translate(tablePunctuation)

    return str(removepunctuation)


# def prepare_df(data_cv, image, filename):
#     # Split data into nested list
#     datalist = list(map(lambda x: x.split('\t'), data_cv.split('\n')))
#
#     print(datalist)
#     df = pd.DataFrame(datalist[1:], columns=datalist[0])
#
#     df.dropna(inplace=True)  # Will drop rows with missing values
#     col_int = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height',
#                'conf']
#     col_float = ['conf']
#     df[col_float] = df[col_float].astype(float)
#     df[col_int] = df[col_int].astype(int)
#
#     print(df.info())
#
#     useFulData = df.query("conf >= 30")
#
#     # Dataframe
#     timesheet = pd.DataFrame()
#     timesheet["text"] = useFulData["text"]
#     timesheet["id"] = filename
#     print(timesheet)
#
#
#     level = "word"
#     for l, x, y, w, h, c, txt in df[['level', 'left', 'top', 'width', 'height', 'conf', 'text']].values:
#         # print(l,x,y,w,h,c)
#         if level == "page":
#             if l == 1:
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             else:
#                 continue
#         elif level == "block":
#             if l == 2:
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             else:
#                 continue
#         elif level == "para":
#             if l == 3:
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             else:
#                 continue
#         elif level == "line":
#             if l == 4:
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
#             else:
#                 continue
#         elif level == "word":
#             if l == 5:
#                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
#                 cv2.putText(image, txt, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
#             else:
#                 continue
#
#     cv2.imshow("bounding_box", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
#
#
#     img_path = glob("timesheets/images/*.jpg")
#
#     img_path = img_path[0]
#     _, filename = os.path.split(img_path)
#
#     img_cv2 = cv2.imread(img_path)
#     image = img_cv2.copy()
#     # cv2.imshow("Timesheet", img_cv2)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # converting image into gray scale image
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # converting it to binary image by Thresholding
#     # this step is require if you have colored image because if you skip this part
#     # then tesseract won't able to detect text correctly and this will give incorrect result
#     threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def entity_parser(text, label):
    if label == "PHONE":
        text = text.lower()
        text = re.sub(r'\D', '', text)
    elif label in ("NAME", "VORNAME"):
        name_replace = ["vorname", "nachname", "vorname ", "nachname ", "dcp-mitarbeiter"]
        text = text.lower()
        for item in name_replace:
            text = text.replace(item, "")
        allow_special_char = "\\-"
        text = re.sub(r'[^A-Za-zäöüÄÖÜß{} ]'.format(allow_special_char), '', text)
        text = text.title()
    elif label in ("LOC"):
        allow_special_char = "\\-"
        text = re.sub(r'[^A-Za-z{} ]'.format(allow_special_char), '', text)
        text = text.title()
    elif label == "EMAIL":
        text = text.lower()
        allow_special_char = "@_.\\-"
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)
    elif label == "WEB":
        text = text.lower()
        allow_special_char = "-#:/.%"
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)
    elif label == "PROJEKT":
        text = text.split()
        if text[0].lower() == "projekt":
            text.pop(0)
        text = " ".join(text)
        text = text.strip()
        allow_special_char = "-#:/.% "
        text = re.sub(r'[^((\w+ )*\w+)?$]'.format(allow_special_char), '', text)
        text = text.title()
    elif label == "POS":
        text = text.lower()
        text = re.sub(r'\D', '', text)
    elif label == "BEST":
        text = text.lower()
        text = text.replace("sapbestellnummer", "")
        allow_special_char = "@_#\\-"
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char), '', text)
    elif label == "PT":
        text = text.lower()
        allow_special_char = ",."
        text = re.sub(r'[^0-9,. ]'.format(allow_special_char), '', text)
    elif label == "DATE":
        text = text.lower()
        allow_special_char = ".\\-"
        text = re.sub(r'[^0-9 ]'.format(allow_special_char), '', text)
        text = text[-6:]
        date = datetime.date(int(text[2:]), int(text[:2]), 1)
        text = date.replace(day=calendar.monthrange(date.year, date.month)[1])
        print(type(text))
        print(text)
    return text


def final_prediction(data_cv, image, original_image):
    # Convert data into dataframe
    datalist = list(map(lambda x: x.split('\t'), data_cv.split('\n')))
    df = pd.DataFrame(datalist[1:], columns=datalist[0])

    # Clean dataframe and extract text only, joined by spaces
    df.dropna(inplace=True)  # Will drop rows with missing values
    df['text'] = df['text'].apply(cleantext)
    data_clean = df.query("text != ''")
    content = " ".join([w for w in data_clean["text"]])

    # Get prediction from NER model
    doc = model_ner(content)

    # Display the entity relations predicted by NER model
    # displacy.serve(doc, style="ent")

    # Tagging for every token in extracted text
    docjson = doc.to_json()
    doc_text = docjson["text"]

    # Creation tokens
    dataframe_tokens = pd.DataFrame(docjson["tokens"])
    dataframe_tokens["token"] = dataframe_tokens[["start", "end"]].apply(lambda x: doc_text[x[0]:x[1]], axis=1)

    right_table = pd.DataFrame(docjson["ents"])[["start", "label"]]
    dataframe_tokens = pd.merge(dataframe_tokens, right_table, how="left", on="start")
    dataframe_tokens.fillna("O", inplace=True)

    # Enrich data_clean dataframe with start and end labels
    data_clean["end"] = data_clean["text"].apply(lambda x: len(x)+1).cumsum()- 1
    data_clean["start"] = data_clean[["text", "end"]].apply(lambda x: x[1] - len(x[0]), axis=1)

    # Innerjoin based on the start positions
    dataframe_info = pd.merge(data_clean, dataframe_tokens[["start", "token", "label"]], how="inner", on="start")
    # print(dataframe_info.tail(20))

    # Change the data type of the relevant columns
    col_int = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf']
    col_float = ['conf']
    # col_str = ["label"]
    dataframe_info[col_float] = dataframe_info[col_float].astype(float)
    dataframe_info[col_int] = dataframe_info[col_int].astype(int)
    # dataframe_info[col_str] = dataframe_info[col_str].astype(str)

    # Optional: Draw bounding boxes with label information
    bb_df = dataframe_info.query("label != 'O' ")
    bb_img = original_image

    for x, y, w, h, c, label in bb_df[['left', 'top', 'width', 'height', 'conf', 'label']].values:
        # x = int(x)
        # y = int(y)
        # w = int(w)
        # h = int(h)

        cv2.rectangle(bb_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(bb_img, str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.putText(bb_img, str(c), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    cv2.imshow("Predictions", bb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Remove B and I tokens from labels in dataframe
    bb_df["label"] = bb_df["label"].apply(lambda x: x[2:])
    # print(bb_df.tail(20))
    # Group labels to combine corresponding values

    bb_df["group"] = bb_df["label"].apply(grp_gen.getgroup)

    # Show bounding boxes based on grouped labels
    bb_df["right"] = bb_df["left"] + bb_df["width"]
    bb_df["bottom"] = bb_df["top"] + bb_df["height"]

    # Tagging by group
    col_group = ["left", "top", "right", "bottom", "label", "token", "group"]
    group_tag_img = bb_df[col_group].groupby(by="group")
    # print(group_tag_img.tail(20))

    img_tagging = group_tag_img.agg({
        "left": min,
        "right": max,
        "top": min,
        "bottom": max,
        "label": np.unique,
        "token": lambda x: " ".join(x)
    })

    # print(img_tagging)


    # Optional: Draw grouped bounding box
    # group_img = original_image
    # for l, r, t, b, label, token in img_tagging.values:
    #
    #     cv2.rectangle(group_img, (l, t), (r, b), (0, 0, 255), 2)
    #     cv2.putText(group_img, label, (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
    #     # cv2.putText(group_img, str(token), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
    #
    # cv2.imshow("Grouped Predictions", group_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Create a Parser for the grouped boxes
    # See separate file parser.py

    # Extract entities
    info_array = dataframe_info[["token", "label"]].values
    entities = dict(NAME=[], VORNAME=[], PROJEKT=[], PT=[], POS=[], BEST=[], LOC=[], DATE=[])

    previous = "O"

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]
        text = entity_parser(token, label_tag)
        if bio_tag in ("B", "I"):
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)
                else:
                    if label_tag in ("PROJEKT"):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text

        previous = label_tag

    return entities, bb_img

# Resizing the images
def resizer(image, width=600):
    # get width and height
    h, w, c = image.shape

    height = int((h/w) * width)
    size = (width, height)
    image = cv2.resize(image, size)
    return image, size


# Image enhancing, eg. sharpen and edge-detection
def image_enhancer_closing(image):
    detail = cv2.detailEnhance(image, sigma_s=20, sigma_r=0.15)
    # Turn into gray-scale image
    gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # edge-detection
    edge_image = cv2.Canny(blur, 75, 200)
    # Morphological transformation
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edge_image, kernel, iterations=1)
    # Closing method used to close any gaps detected in image
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    return closing


def contour_finder(image_find, image_draw):
    contours, hierarchy = cv2.findContours(image_find, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*peri, True)
        if len(approx) == 4:
            four_points = np.squeeze(approx)
            break
    return cv2.drawContours(image_draw, [four_points], -1, (0, 255, 0), 3), four_points


# find four points for original (!) image
def crop_image(image, four_points, size):
    multiplier = image.shape[1] / size[0]
    four_points_original = four_points * multiplier
    four_points_original = four_points_original.astype(int)
    return four_points_original


def apply_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def plot_images_inline(image1, image2):
    # plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title("Image 1")
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title("Image 2")
    plt.show()


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

# test_path = r'C:\Users\bbarenthien\PycharmProjects\DocuScanner\Version_2\timesheettest'
# for filename in listdir_fullpath(test_path):
#     im_new, original_image = train_process_image_for_ocr(filename)


def new_final_prediction(imgPaths):
    # Convert data into dataframe
    for img_path in tqdm(imgPaths, desc="TimeSheet"):
        datalist = []
        # img_path = img_path[image]
        _, filename = os.path.split(img_path)

        img_cv2 = cv2.imread(img_path)
        image = img_cv2.copy()
        data_cv = reader.readtext(img_path)
        # print(data_cv)

        # Split data into nested list loop over the results
        for (bbox, text, prob) in data_cv:
            # display the OCR'd text and associated probability
            # print("[INFO] {:.4f}: {}".format(prob, text))
            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))

            # cleanup the text and draw the box surrounding the text along
            # with the OCR'd text itself
            # text = cleanup_text(text)
            list_entry = [tl, tr, br, bl, text, prob]
            cv2.rectangle(image, tl, br, (0, 255, 0), 2)
            cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            datalist.append(list_entry)
            # datalist = list(map(lambda x: x.split('\t'), data_cv.split('\n')))

        # print(datalist)
        df_columns = ['bbox', 'text', 'conf']
        df_columns_ext = ['bb_tl', 'bb_tr', 'bb_br', 'bb_bl', 'text', 'conf']

        # print(datalist)
        # df = pd.DataFrame(data_cv, columns=df_columns)
        #
        # print(df.info())

        df = pd.DataFrame(datalist, columns=df_columns_ext)

        # print(df.info())

        df.dropna(inplace=True)  # Will drop rows with missing values
        # print(df.head(20))

        # Clean dataframe and extract text only, joined by spaces
        # df.dropna(inplace=True)  # Will drop rows with missing values
        df['text'] = df['text'].apply(cleantext)
        data_clean = df.query("text != ''")
        # print(data_clean)
        # content = " ".join([w for w in data_clean["text"]])
        content = " ".join([w for w in data_clean["text"]])
        # print(content)
        # Get prediction from NER model
        doc = model_ner(content)

        # print(doc)

        # Display the entity relations predicted by NER model
        # displacy.serve(doc, style="ent")

        # Tagging for every token in extracted text
        docjson = doc.to_json()
        # print(docjson)
        doc_text = docjson["text"]
        # print(f"Hello \n {doc_text}")

        # Creation tokens
        dataframe_tokens = pd.DataFrame(docjson["tokens"])
        # print(dataframe_tokens)
        dataframe_tokens["token"] = dataframe_tokens[["start", "end"]].apply(lambda x: doc_text[x[0]:x[1]], axis=1)
        # print(dataframe_tokens["token"])

        right_table = pd.DataFrame(docjson["ents"])[["start", "label"]]
        # print(right_table)
        dataframe_tokens = pd.merge(dataframe_tokens, right_table, how="left", on="start")
        dataframe_tokens.fillna("O", inplace=True)
        # print(dataframe_tokens.head(40))

        # Enrich data_clean dataframe with start and end labels
        # data_clean["end"] = data_clean["text"].apply(lambda x: len(x)+1).cumsum() - 1
        # data_clean["start"] = data_clean[["text", "end"]].apply(lambda x: x[1] - len(x[0]), axis=1)
        #
        # # Innerjoin based on the start positions
        # dataframe_info = pd.merge(data_clean, dataframe_tokens[["start", "token", "label"]], how="inner", on="start")
        # print(dataframe_info.head(30))

        # Change the data type of the relevant columns
        # col_int = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf']
        # col_float = ['conf']
        # # col_str = ["label"]
        # dataframe_info[col_float] = dataframe_info[col_float].astype(float)
        # dataframe_info[col_int] = dataframe_info[col_int].astype(int)
        # dataframe_info[col_str] = dataframe_info[col_str].astype(str)

        # # Optional: Draw bounding boxes with label information
        bb_df = dataframe_tokens.query("label != 'O' ")
        bb_img = img_cv2

        # for x, y, label in bb_df[['start', 'end', 'label']].values:
        #     x = int(x)
        #     y = int(y)
        #     w = int(y - x)
        #     h = 5
        #
        #     cv2.rectangle(bb_img, (x, y), (w, h), (0, 0, 255), 2)
        #     cv2.putText(bb_img, str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        #     # cv2.putText(bb_img, str(c), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        #
        # cv2.imshow("Predictions", bb_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Remove B and I tokens from labels in dataframe
        bb_df["label"] = bb_df["label"].apply(lambda x: x[2:])
        # print(bb_df.tail(20))
        # Group labels to combine corresponding values

        bb_df["group"] = bb_df["label"].apply(grp_gen.getgroup)

        # # Show bounding boxes based on grouped labels
        # bb_df["right"] = bb_df["left"] + bb_df["width"]
        # bb_df["bottom"] = bb_df["top"] + bb_df["height"]

        # Tagging by group
        # col_group = ["left", "top", "right", "bottom", "label", "token", "group"]
        # group_tag_img = bb_df[col_group].groupby(by="group")
        # print(group_tag_img.tail(20))

        # img_tagging = group_tag_img.agg({
        #     "left": min,
        #     "right": max,
        #     "top": min,
        #     "bottom": max,
        #     "label": np.unique,
        #     "token": lambda x: " ".join(x)
        # })

        # print(img_tagging)


        # Optional: Draw grouped bounding box
        # group_img = original_image
        # for l, r, t, b, label, token in img_tagging.values:
        #
        #     cv2.rectangle(group_img, (l, t), (r, b), (0, 0, 255), 2)
        #     cv2.putText(group_img, label, (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        #     # cv2.putText(group_img, str(token), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        #
        # cv2.imshow("Grouped Predictions", group_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Create a Parser for the grouped boxes
        # See separate file parser.py

        # Extract entities
        info_array = dataframe_tokens[["token", "label"]].values
        print(info_array)
        entities = dict(NAME=[], VORNAME=[], PROJEKT=[], PT=[], POS=[], BEST=[], LOC=[], DATE=[])

        previous = "O"

        for token, label in info_array:
            bio_tag = label[0]
            label_tag = label[2:]
            text = entity_parser(token, label_tag)
            if bio_tag in ("B", "I"):
                if previous != label_tag:
                    entities[label_tag].append(text)
                else:
                    if bio_tag == "B":
                        entities[label_tag].append(text)
                    else:
                        if label_tag in ("PROJEKT"):
                            print(entities[label_tag][-1])
                            entities[label_tag][-1] = (entities[label_tag][-1] + " " + text).strip()
                            print(entities[label_tag][-1])
                        else:
                            entities[label_tag][-1] = entities[label_tag][-1] + text

            previous = label_tag
        print(entities)
        total_pt, pt_remote = onsite_calculator(entities)
        print(f"Total PT: {total_pt}")
        print(f"PT remote: {pt_remote}")
    # return entities, bb_img
    return bb_img, entities


def onsite_calculator(entities):
    remote_data = entities["LOC"]
    days_onsite = 0
    remote_strings = ["Ons", "Fran"]
    for onsite in remote_strings:
        # remote_str = "Ons"
        days_onsite = days_onsite + int(len([i for i in remote_data if onsite in i]))
    total_pt = entities["PT"][0].replace(",", ".")
    pt_onsite = float(total_pt) - float(days_onsite)
    return total_pt, pt_onsite


