import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output
import os


def merge_content(lines_ct: list):
        content = ""
        for line in lines_ct:
            for word in line:
                content += word + ' '
        return content
    
    
def main(path):
    folder_names = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    img_names = [f for f in os.listdir(path) if f.find('jpg') != -1 or f.find('.png') != -1]
    img_paths = [os.path.join(path, f) for f in img_names]
    #write into txt file
    with open('./imgs_crop_data.txt', 'w', encoding='utf-8') as f:
        #img
        for path_id, im_path in enumerate(img_paths):
            img = cv2.imread(im_path)

            text_infor = pytesseract.image_to_data(Image.fromarray(img), lang='vie', output_type=Output.DICT)
            remove_ids = []
            for id, conf in enumerate(text_infor['conf']):
                if conf  == '-1' or conf < 40:
                    remove_ids.append(id)
            for key in text_infor.keys():
                for id, val in enumerate(remove_ids):
                    del text_infor[key][val - id]
            extract_ids = {}
            for ch_id in range(text_infor['block_num'].__len__()):
                if not bool(extract_ids):
                    extract_ids = {1: {1: {1: [ch_id]}}}
                else:
                    max_block_num = (list(extract_ids.keys()))[-1]
                    max_par_num = (list(extract_ids[max_block_num].keys()))[-1]
                    max_line_num = (list(extract_ids[max_block_num][max_par_num].keys()))[-1]
                    if text_infor['block_num'][ch_id] != max_block_num:
                        extract_ids[max_block_num + 1] = {1: {1: [ch_id]}}
                    else: 
                        if text_infor['par_num'][ch_id] != max_par_num:
                            extract_ids[max_block_num][max_par_num + 1] = {1: [ch_id]}
                        else:
                            if text_infor['line_num'][ch_id] != max_line_num:
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
                        b_x, b_y, b_w, b_h = text_infor['left'][begin_id], text_infor['top'][begin_id], text_infor['width'][begin_id], text_infor['height'][begin_id]
                        e_x, e_y, e_w, e_h = text_infor['left'][end_id], text_infor['top'][end_id], text_infor['width'][end_id], text_infor['height'][end_id]
                        x1, y1, x2, y2 = b_x, min(b_y, e_y), e_x + e_w, max(b_y + b_h, e_y, e_h)
                        pr_lines_coords.append([x1, y1, x2, y2])
                        line_ct = []
                        for word_idx in extract_ids[bl][par][line]:
                            line_ct.append(text_infor['text'][word_idx])
                        pr_lines_ct.append(line_ct)
                    paragraphs_infor.append([pr_lines_ct, pr_lines_coords])
            for para in paragraphs_infor:
                f.write(merge_content(para[0])+"\n") 
            if path_id != len(img_paths) - 1: 
                f.write("\n")
        #folder
        for path_id2, name in enumerate(folder_names):
            dir_path = os.path.join(path, name)
            file_names = os.listdir(dir_path)
            for sub_name in file_names:
                sub_path = os.path.join(dir_path, sub_name)
                img = cv2.imread(sub_path)
                text_infor = pytesseract.image_to_data(Image.fromarray(img), lang='vie', output_type=Output.DICT)
                remove_ids = []
                for id, conf in enumerate(text_infor['conf']):
                    if conf  == '-1' or conf < 40:
                        remove_ids.append(id)
                for key in text_infor.keys():
                    for id, val in enumerate(remove_ids):
                        del text_infor[key][val - id]
                extract_ids = {}
                for ch_id in range(text_infor['block_num'].__len__()):
                    if not bool(extract_ids):
                        extract_ids = {1: {1: {1: [ch_id]}}}
                    else:
                        max_block_num = (list(extract_ids.keys()))[-1]
                        max_par_num = (list(extract_ids[max_block_num].keys()))[-1]
                        max_line_num = (list(extract_ids[max_block_num][max_par_num].keys()))[-1]
                        if text_infor['block_num'][ch_id] != max_block_num:
                            extract_ids[max_block_num + 1] = {1: {1: [ch_id]}}
                        else: 
                            if text_infor['par_num'][ch_id] != max_par_num:
                                extract_ids[max_block_num][max_par_num + 1] = {1: [ch_id]}
                            else:
                                if text_infor['line_num'][ch_id] != max_line_num:
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
                            b_x, b_y, b_w, b_h = text_infor['left'][begin_id], text_infor['top'][begin_id], text_infor['width'][begin_id], text_infor['height'][begin_id]
                            e_x, e_y, e_w, e_h = text_infor['left'][end_id], text_infor['top'][end_id], text_infor['width'][end_id], text_infor['height'][end_id]
                            x1, y1, x2, y2 = b_x, min(b_y, e_y), e_x + e_w, max(b_y + b_h, e_y, e_h)
                            pr_lines_coords.append([x1, y1, x2, y2])
                            line_ct = []
                            for word_idx in extract_ids[bl][par][line]:
                                line_ct.append(text_infor['text'][word_idx])
                            pr_lines_ct.append(line_ct)
                        paragraphs_infor.append([pr_lines_ct, pr_lines_coords])
                for para in paragraphs_infor:
                    f.write(merge_content(para[0])+"\n") 
            if path_id2 != len(folder_names) - 1:
                f.write("\n")
                

if __name__ == '__main__':
    imgs_dir = './crop_imgs/'
    main(imgs_dir)
    