import settings
from imutils.perspective import four_point_transform
import numpy as np
import cv2
import pandas as pd
import os
from pdf2image import convert_from_path
import re
import string
import spacy
import easyocr
import calendar
import datetime


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


def save_upload_image(fileObj):
    filename = fileObj.filename

    name, ext = filename.rsplit(".", 1)
    if ext == "pdf":
        save_filename = "upload." + ext
    else:
        save_filename = "upload." + ext
    upload_img_path = settings.join_path(settings.SAVE_DIR, save_filename)
    fileObj.save(upload_img_path)
    if ext == "pdf":
        filename = "upload.jpg"
        upload_img_path = settings.join_path(settings.SAVE_DIR, filename)
        image_directory = os.path.dirname(upload_img_path)
        images = convert_from_path(image_directory + '/' + save_filename)
        for i in range(len(images)):
            images[i].save(upload_img_path, 'JPEG')
    return upload_img_path


def array_to_json(numpy_array):
    points = []
    for pt in numpy_array.tolist():
        points.append({'x': pt[0], 'y': pt[1]})
    return points


class DocumentScan:
    def __init__(self):
        pass


    @staticmethod
    def resizer(image, width=500):
        # get widht and height
        h, w, c = image.shape

        height = int((h / w) * width)
        size = (width, height)
        image = cv2.resize(image, (width, height))
        return image, size

    @staticmethod
    def apply_brightness_contrast(input_img, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf


    # def new_convert_pdfs(self, image_path):
    #     print(image_path)
    #     filename = "upload.jpg"
    #     CONVERTED_IMAGE_PATH = settings.join_path(settings.MEDIA_DIR, filename)
    #
    #     image = cv2.imread(image_path)
    #     # image_directory = Path(image_path)
    #     image_directory = os.path.dirname(image_path)
    #
    #     abs_image_path = image_directory
    #
    #     for file in os.listdir(abs_image_path):
    #         if file.endswith("pdf"):
    #             images = convert_from_path(abs_image_path + '/' + file)
    #             # stem = Path(file).stem
    #             for i in range(len(images)):
    #                 # file = os.path.join(folder + '\\jpegs\\', stem)
    #                 # file = os.path.join(image_path, stem)
    #                 images[i].save(CONVERTED_IMAGE_PATH, 'JPEG')
    #     return CONVERTED_IMAGE_PATH
    #         # else:
    #         #     print(CONVERTED_IMAGE_PATH)
    #         #     cv2.imwrite(CONVERTED_IMAGE_PATH, image)

    def document_scanner(self, image_path):
        four_points = None
        self.image = cv2.imread(image_path)
        img_re, self.size = self.resizer(self.image)
        filename = 'resize_image.jpg'
        RESIZE_IMAGE_PATH = settings.join_path(settings.MEDIA_DIR, filename)

        cv2.imwrite(RESIZE_IMAGE_PATH, img_re)

        try:

            detail = cv2.detailEnhance(img_re, sigma_s=20, sigma_r=0.15)
            gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)  # GRAYSCALE IMAGE
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # edge detect
            edge_image = cv2.Canny(blur, 75, 200)
            # morphological transform
            kernel = np.ones((5, 5), np.uint8)
            dilate = cv2.dilate(edge_image, kernel, iterations=1)
            closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

            # find the contours
            contours, hire = cv2.findContours(closing,
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)

            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    four_points = np.squeeze(approx)
                    break
            print(four_points)
            return four_points, self.size

        except:
            return None, self.size


    def calibrate_to_original_size(self, four_points):
        # find four points for original image

        multiplier = self.image.shape[1] / self.size[0]
        four_points_orig = four_points * multiplier
        four_points_orig = four_points_orig.astype(int)
        wrap_image = four_point_transform(self.image, four_points_orig)
        # apply magic color to wrap image
        magic_color = self.apply_brightness_contrast(wrap_image, brightness=0, contrast=0)

        return magic_color

def cleantext(txt):
    whitespace = string.whitespace
    punctuation = '|!"#$%()*+/:;<=>?@[\\]^_`{|}~'
    tableWhitespace = str.maketrans('', '', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    # text = text.lower()
    # removewhitespace = text.translate(tableWhitespace)
    removepunctuation = text.translate(tablePunctuation)

    return str(removepunctuation)


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
        print(text)
        if text[0].lower() == "projekt":
            text.pop(0)
        text = " ".join(text)
        print(text)
        text = text.strip()
        print(text)
        allow_special_char = "-#:/.% \\-"
        text = re.sub(r'[^((\w+ \-)*\w+)?$]'.format(allow_special_char), '', text)
        print(text)
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
        try:
            date = datetime.date(int(text[2:]), int(text[:2]), 1)
            text = date.replace(day=calendar.monthrange(date.year, date.month)[1])
            return text
        except:
            today = datetime.datetime.now()
            text = datetime.date(today.year, today.month-1, calendar.monthrange(today.year, today.month-1)[1])
            return text
    return text


def new_final_prediction(img_path):
    # Convert data into dataframe
# for img_path in tqdm(imgPaths, desc="TimeSheet"):
    datalist = []
    # img_path = img_path[image]
    _, filename = os.path.split(img_path)

    img_cv2 = cv2.imread(img_path)
    image = img_cv2.copy()
    data_cv = reader.readtext(image)
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
    return entities


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
#
# def save_upload_image(fileObj):
#     filename = fileObj.filename
#     name, ext = filename.split('.')
#     save_filename = 'upload.' + ext
#     upload_image_path = settings.join_path(settings.SAVE_DIR,
#                                            save_filename)
#
#     fileObj.save(upload_image_path)
#
#     return upload_image_path
#
#
# def array_to_json(numpy_array):
#     points = []
#     for pt in numpy_array.tolist():
#         points.append({'x': pt[0], 'y': pt[1]})
#
#     return points
#
#
# class DocumentScan():
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def resizer(image, width=250):
#         # get widht and height
#         h, w, c = image.shape
#
#         height = int((h / w) * width)
#         size = (width, height)
#         image = cv2.resize(image, (width, height))
#         return image, size
#
#     @staticmethod
#     def apply_brightness_contrast(input_img, brightness=0, contrast=0):
#
#         if brightness != 0:
#             if brightness > 0:
#                 shadow = brightness
#                 highlight = 255
#             else:
#                 shadow = 0
#                 highlight = 255 + brightness
#             alpha_b = (highlight - shadow) / 255
#             gamma_b = shadow
#
#             buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
#         else:
#             buf = input_img.copy()
#
#         if contrast != 0:
#             f = 131 * (contrast + 127) / (127 * (131 - contrast))
#             alpha_c = f
#             gamma_c = 127 * (1 - f)
#
#             buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
#
#         return buf
#
#     def document_scanner(self, image_path):
#         self.image = cv2.imread(image_path)
#         img_re, self.size = self.resizer(self.image)
#         filename = 'resize_image.jpg'
#         RESIZE_IMAGE_PATH = settings.join_path(settings.MEDIA_DIR, filename)
#
#         cv2.imwrite(RESIZE_IMAGE_PATH, img_re)
#
#         try:
#
#             detail = cv2.detailEnhance(img_re, sigma_s=20, sigma_r=0.15)
#             gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)  # GRAYSCALE IMAGE
#             blur = cv2.GaussianBlur(gray, (5, 5), 0)
#             # edge detect
#             edge_image = cv2.Canny(blur, 75, 200)
#             # morphological transform
#             kernel = np.ones((5, 5), np.uint8)
#             dilate = cv2.dilate(edge_image, kernel, iterations=1)
#             closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
#
#             # find the contours
#             contours, hire = cv2.findContours(closing,
#                                               cv2.RETR_LIST,
#                                               cv2.CHAIN_APPROX_SIMPLE)
#
#             contours = sorted(contours, key=cv2.contourArea, reverse=True)
#             for contour in contours:
#                 peri = cv2.arcLength(contour, True)
#                 approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
#
#                 if len(approx) == 4:
#                     four_points = np.squeeze(approx)
#                     break
#
#             return four_points, self.size
#
#         except:
#             return None, self.size
#
#     def calibrate_to_original_size(self, four_points):
#         # find four points for original image
#
#         multiplier = self.image.shape[1] / self.size[0]
#         four_points_orig = four_points * multiplier
#         four_points_orig = four_points_orig.astype(int)
#         wrap_image = four_point_transform(self.image, four_points_orig)
#         # apply magic color to wrap image
#         magic_color = self.apply_brightness_contrast(wrap_image, brightness=40, contrast=60)
#
#         return magic_color