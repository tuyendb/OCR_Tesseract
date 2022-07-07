import numpy as np
import cv2
from pdf2image import convert_from_bytes
import pytesseract
from pytesseract import Output
import os 
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from objects.line import Line
from objects.paragraph import Paragraph


#######################CONVERT PDF FILE TO PAGE IMGS########################
def pdf2jpg(pdf_path, first_page, last_page):
    images = convert_from_bytes(pdf_path, grayscale=True, jpegopt=True,
                               first_page=first_page, last_page=last_page)
    book = []
    for page in images:
        page = page.convert('RGB')
        open_cv_image = np.array(page)
        # Convert RGB to BGR
        page = open_cv_image[:, :, ::-1].copy()
        book.append(page)
    return book
###########################################################################

######### EXPAND BOX AFTER DETECTION ######
def expand_box(box):
    x1, y1, x2, y2 = box
    return [x1-20, y1, x2+20, y2]
###########################################


##################### CLASS DETECTION BLOCKS(TITLE, FIGURE, ...) #################################
class TextBlockDetection:

    def __init__(self, page_img, text_block_detector):
        self.page_img = page_img
        self.text_block_detector = text_block_detector
        self._text_block_det = None
        self._reduced_tbs = None

    @property
    def text_block_detection(self):
        if self._text_block_det is None:
            outputs = self.text_block_detector(self.page_img)
            #Debug outputs
            class_labels = ['text', 'title', 'list', 'table', 'figure']
            instances = outputs["instances"].to('cpu')
            pred_boxes = instances.pred_boxes
            scores = instances.scores
            pred_classes = instances.pred_classes
            boxes = []
            selected_labels = ['title', 'figure']
            for i in range(0, len(pred_boxes)):
                box = pred_boxes[i].tensor.numpy()[0]
                box = list(box.astype(int))
                score = round(float(scores[i].numpy()), 2)
                label_key = int(pred_classes[i].numpy())
                label = class_labels[label_key]
                if label in selected_labels:
                    x1, y1, x2, y2 = expand_box(box)
                    boxes.append([[x1, y1, x2, y2], label, score, self.page_img[y1:y2, x1:x2]])
            self._text_block_det = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
        return self._text_block_det
################################################################################################


################## REMOVE FIGURES FROM PAGE ###################
def check_limited_area(bounding_box, shape):
    x1, y1, x2, y2 = bounding_box
    area = (y2 - y1)*(x2 - x1)
    remove = None
    if area > shape[0]*shape[1]/100:
        remove = True
    else:
        remove = False
    return remove


def remove_figure(img, figure_coords):
    shape = img.shape
    if check_limited_area(figure_coords, shape):
        x1, y1, x2, y2 = figure_coords
        remove_area = img[y1:y2, x1:x2]
        mask = np.ones_like(remove_area.shape, np.uint8)
        img[y1:y2, x1:x2] = cv2.bitwise_not(mask, mask)
    else:
        pass
    return img
###############################################################


################### CROP LINE FROM TEXTBLOCK(TITLE, FIGURE) #######################################################################################################################################
class CropLineImgFromTextBlock:

    def __init__(self, block_img):
        self._block_img = block_img
        self._text_info = None
        self._lines_infor = None

    @property
    def text_info_extract(self):
        if self._text_info is None:
            self._text_info = pytesseract.image_to_data(
                cv2.cvtColor(self._block_img, cv2.COLOR_BGR2RGB), 
                lang='vie', 
                output_type=Output.DICT
                )
            remove_ids = []
            for id, text in enumerate(self._text_info['text']):
                if text  == "":
                    remove_ids.append(id)
            for key in self._text_info.keys():
                for id, val in enumerate(remove_ids):
                    if id == 0:
                        del self._text_info[key][val]
                    else:
                        del self._text_info[key][val - id]
        return self._text_info

    @property
    def lines_infor(self):  
        if self._lines_infor is None:
            setted_num_line = list(set(self._text_info['line_num']))
            line_word_coords = []
            word_id_line = {}
            for line_num in setted_num_line:
                word_id_line[line_num] = []
                for i, num in enumerate(self._text_info['line_num']):
                    if num == line_num:
                        word_id_line[line_num].append(i)
            for key in word_id_line.keys():
                begin_id = word_id_line[key][0]
                end_id = word_id_line[key][-1]
                b_x, b_y, b_w, b_h = self._text_info['left'][begin_id], self._text_info['top'][begin_id], self._text_info['width'][begin_id], self._text_info['height'][begin_id]
                e_x, e_y, e_w, e_h = self._text_info['left'][end_id], self._text_info['top'][end_id], self._text_info['width'][end_id], self._text_info['height'][end_id]
                line_word_coords.append([(b_x, b_y, b_w, b_h), (e_x, e_y, e_w, e_h)])
            lines_crop = []
            lines_coords = []
            for line_coord in line_word_coords:
                x1 = line_coord[0][0]
                y1 = min(line_coord[0][1], line_coord[1][1])
                x2 = line_coord[1][0] + line_coord[1][2]
                y2 = max((line_coord[0][1] + line_coord[0][3]), (line_coord[1][1] + line_coord[1][3]))
                line_img_crop = self._block_img[y1:y2, x1:x2]
                lines_crop.append(line_img_crop)                    #imgs_crop
                lines_coords.append([x1, y1, x2, y2])               #crop_coords
                self._lines_infor = [lines_crop, lines_coords]
            return self._lines_infor
##############################################################################################################################################################################################


########################CROP IMGS BELONG TO TITLE########################
def crop_imgs_between_2_titles(pages, title1, title2):
    page_id1, tb1 = title1
    page_id2, tb2 = title2
    y2_tt1 = tb1[0][3]
    y1_tt2 = tb2[0][1]
    crop_imgs = []
    if page_id1 == page_id2:
        if y2_tt1 < y1_tt2:
            page_width = pages[page_id2].shape[1]
            crop = pages[page_id1][y2_tt1:y1_tt2, 0:page_width]
            crop_imgs.append(crop)
        else:
            pass
    else:
        page1_height, page1_width = pages[page_id1].shape[:2]
        page2_height, page2_width = pages[page_id2].shape[:2]
        crop1 = pages[page_id1][y2_tt1:page1_height, 0:page1_width]
        crop2 = pages[page_id2][0:y1_tt2, 0:page2_width]
        crop_imgs.append(crop1)
        if page_id1 == page_id2 - 1:
            pass
        else:
            mesial_page_count = page_id2 - page_id1
            for i in range(1, mesial_page_count):
                crop_imgs.append(pages[page_id1 + i])
        crop_imgs.append(crop2)
    return crop_imgs

    
def crop_imgs_final(pages, final_title):
    page_id, tb = final_title
    y2_fn_tt = tb[0][3]
    crop_imgs = []
    for page_idx in range(page_id, len(pages)):
        if page_idx == page_id:
            page_height, page_width = pages[page_idx].shape[:2]
            crop1 = pages[page_idx][y2_fn_tt:page_height, 0:page_width]
            crop_imgs.append(crop1)
        else:
            crop_imgs.append(pages[page_idx])
    return crop_imgs
##########################################################################


############################## EXTRACT PARAGRAPHS INFORMATION FROM TESSERACT ##############################################################################
def extract_paragr_infor_from_tesseract(img):
    text_info = pytesseract.image_to_data(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
        lang='vie', 
        output_type=Output.DICT, 
        )
    remove_ids = []
    for id, text in enumerate(text_info['text']):
        if text  == "":
            remove_ids.append(id)
    for key in text_info.keys():
        for id, val in enumerate(remove_ids):
            if id == 0:
                del text_info[key][val]
            else:
                del text_info[key][val - id]
    extract_ids = {}
    for ch_id in range(text_info['block_num'].__len__()):
        if not bool(extract_ids):
            extract_ids = {1: {1: {1: [ch_id]}}}
        else:
            max_block_num = (list(extract_ids.keys()))[-1]
            max_par_num = (list(extract_ids[max_block_num].keys()))[-1]
            max_line_num = (list(extract_ids[max_block_num][max_par_num].keys()))[-1]
            if text_info['block_num'][ch_id] != max_block_num:
                extract_ids[max_block_num + 1] = {1: {1: [ch_id]}}
            else: 
                if text_info['par_num'][ch_id] != max_par_num:
                    extract_ids[max_block_num][max_par_num + 1] = {1: [ch_id]}
                else:
                    if text_info['line_num'][ch_id] != max_line_num:
                        extract_ids[max_block_num][max_par_num][max_line_num + 1] = [ch_id]
                    else:
                        extract_ids[max_block_num][max_par_num][max_line_num].append(ch_id)
    paragraphs_infor = []
    for bl in extract_ids.keys():
        for par in extract_ids[bl].keys():
            pr_lines_ct = []
            pr_lines_coords = []
            for line in extract_ids[bl][par].keys():
                begin_id = extract_ids[bl][par][line][0]
                end_id = extract_ids[bl][par][line][-1]
                b_x, b_y, b_w, b_h = text_info['left'][begin_id], text_info['top'][begin_id], text_info['width'][begin_id], text_info['height'][begin_id]
                e_x, e_y, e_w, e_h = text_info['left'][end_id], text_info['top'][end_id], text_info['width'][end_id], text_info['height'][end_id]
                x1, y1, x2, y2 = b_x, min(b_y, e_y), e_x + e_w, max(b_y + b_h, e_y, e_h)
                pr_lines_coords.append([x1, y1, x2, y2])
                line_ct = []
                for word_idx in extract_ids[bl][par][line]:
                    line_ct.append(text_info['text'][word_idx])
                pr_lines_ct.append(line_ct)
            paragraphs_infor.append([pr_lines_ct, pr_lines_coords])
    group_ids = {}
    for para_id, para_infor in enumerate(paragraphs_infor):
        if len(group_ids) == 0:
            group_ids[0] = {
                'std_coord': Paragraph(img, para_infor).paragraph_coord,
                'para_ids' : [para_id]
            }
                
        else:
            keys_list = list(group_ids.keys())
            for key in keys_list[::-1]:            
                considered_coord = Paragraph(img, para_infor).paragraph_coord
                if group_ids[key]['std_coord'][0] > considered_coord[2] or group_ids[key]['std_coord'][2] < considered_coord[0]:
                    group_ids[keys_list[-1] + 1] = {
                        'std_coord': Paragraph(img, para_infor).paragraph_coord,
                        'para_ids' : [para_id]
                    }
                else:
                    group_ids[key]['para_ids'].append(para_id)
                    group_ids[key]['std_coord'] = [
                        min(group_ids[key]['std_coord'][0], considered_coord[0]),
                        min(group_ids[key]['std_coord'][1], considered_coord[1]),
                        max(group_ids[key]['std_coord'][2], considered_coord[2]),
                        max(group_ids[key]['std_coord'][3], considered_coord[3]),
                    ]
                    break

    sorted_group_ids = {k: v for k, v in sorted(group_ids.items(), key=lambda item: item[1]['std_coord'][1])}
    new_paragraphs_infor = []
    new_key_list = list(sorted_group_ids.keys())
    for key in new_key_list:
        for new_para_id in sorted_group_ids[key]['para_ids']:
            new_paragraphs_infor.append(paragraphs_infor[new_para_id])
    return new_paragraphs_infor
###############################################################################################################################################################


############################# CHECK, REMOVE UNNECESSARY CONTENT ##############################
def check_lenght(text):
    check_len = len(text) <= 7
    return check_len


def check_uncsr_text(text):
    ntr = False
    check_list = ["Hình", "hình", "Hinh", "hinh"] 
    if len(text) >= 6:
        for check_it in check_list:
            if check_it in text and text[5].isdecimal():
                ntr = True
                break
    return ntr

def check_remove(text):
    check_rm = check_lenght(text) or check_uncsr_text(text)
    return check_rm 
###############################################################################################


def check_outlier(text):
    outlier = False
    text2list = text.split(' ')
    if text2list[0] == '({})'.format(text2list[0][1:-1]) and text2list[0][1:-1].isdecimal():
        outlier = True
    return outlier


################# MERGE PARAGRAPHS INTO 1 PARAGRAPH LOGICALLY ##################
class CheckConditionsToMergePars:

    def __init__(self, page_img1, page_img2, par_infor1 , par_infor2):
        self._paragr1 = Paragraph(page_img1, par_infor1)
        self._paragr2 = Paragraph(page_img2, par_infor2)

    @staticmethod
    def find_tab_distance(bg_line: Line, sc_line: Line):
        tap_distance = abs(sc_line.x_cooords[0] - bg_line.x_cooords[0])
        tap_distance = max(10, tap_distance)
        return tap_distance

    @staticmethod
    def check_not_subtitle(text):
        check_list = ['/', ')', '.', ':', '-', '?']
        is_subtitle = False
        for check in check_list:
            if check in text[:2]:
                is_subtitle = True
                break
        return not is_subtitle

    @property
    def ntm_2pars(self):
        ntm = False
        ft_line_pp = Line(
            self._paragr1._img,
            self._paragr1.lines_coords[0],
            self._paragr1.lines_ct[0]
        )

        ft_line_np = Line(
            self._paragr2._img,
            self._paragr2.lines_coords[0],
            self._paragr2.lines_ct[0]
        )
        if len(self._paragr1.lines_coords) > 1:
            nft_line_pp = Line(
                self._paragr1._img,
                self._paragr1.lines_coords[1],
                self._paragr1.lines_ct[1]
            )
            tab_distance = self.find_tab_distance(ft_line_pp, nft_line_pp)
        else:
            tab_distance = 10
        if ft_line_np.line_content[0].islower() and self.check_not_subtitle(ft_line_np.line_content):
            ntm = True
        else:
            if ft_line_np.x_cooords[0] + 0.5*tab_distance <= ft_line_pp.x_cooords[0]:
                ntm = True
            else:
                pass
        return ntm