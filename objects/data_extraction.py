import os 
import uuid
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from utils.util import *
from objects.paragraph import Paragraph


class DataExtraction:
    def __init__(self, img_crops):
        self._img_crops = img_crops
        self._page_count = img_crops.__len__()
        self._extracted_infor = None

    @property
    def extracted_infor(self):
        if self._extracted_infor is None:
            ct_infor = {}
            for ct_id, ct_crop in enumerate(self._img_crops):
                try:
                    ct_infor[ct_id] = [
                        ct_crop,
                        extract_paragr_infor_from_tesseract(ct_crop)
                    ]
                except:
                    file_name = '{}.jpg'.format(uuid.uuid4())
                    file_path = os.path.join('./saved_imgs/', file_name)
                    cv2.imwrite(file_path, ct_crop)
                    ct_crop = cv2.imread(file_path)
                    ct_infor[ct_id] = [
                        ct_crop,
                        extract_paragr_infor_from_tesseract(ct_crop)
                    ]
                    os.remove(file_path)
            total_paragraph = {}
            key_list = list(ct_infor.keys())
            rm_total = []
            outlier = []
            for key in key_list:
                total_paragraph[key] = []
                img_crop, paragrs_infor = ct_infor[key]
                if len(paragrs_infor) == 0:
                    pass
                else:
                    rm_para = []
                    for para_id, para_infor in enumerate(paragrs_infor):
                        para_ct = Paragraph(img_crop, para_infor).paragraph_content
                        if not check_outlier(para_ct):
                            total_paragraph[key].append(para_ct)
                        else:
                            outlier.append(para_ct)
                            rm_para.append(para_id)
                    if len(rm_para) != 0:
                        rm_total.append([key, rm_para])
            if len(rm_total) != 0:
                for rm in rm_total:
                    for id, val in enumerate(rm[1]):
                        del ct_infor[rm[0]][1][val- id]
            self._extracted_infor = (ct_infor, total_paragraph, outlier)
        return self._extracted_infor

    @property
    def extracted_content(self) -> dict:
        ct_infor, total_paragraphs, outlier = self.extracted_infor
        paragr_content = {}
        if self._page_count >= 2:
            for page_id in range(self._page_count - 1):
                if len(total_paragraphs[page_id]) == 0:
                    pass
                else:
                    if len(total_paragraphs[page_id]) == 1:
                        pass
                    else:
                        para_list = total_paragraphs[page_id]
                        for par_id in range(len(para_list) -1):
                            page_infor = ct_infor[page_id]
                            sp_para_infor1 = page_infor[1][par_id]
                            sp_para_infor2 = page_infor[1][par_id + 1]
                            same_img = page_infor[0]
                            if CheckConditionsToMergePars(same_img, same_img, sp_para_infor1, sp_para_infor2).ntm_2pars:
                                total_paragraphs[page_id][par_id], total_paragraphs[page_id][par_id + 1] = "", para_list[par_id] + para_list[par_id + 1]
                            else:
                                pass
                    if len(total_paragraphs[page_id + 1]) == 0:
                        pass 
                    else:
                        pr_list1 = total_paragraphs[page_id]
                        pr_list2 = total_paragraphs[page_id + 1]
                        page1_infor = ct_infor[page_id]
                        page_img1 = page1_infor[0]
                        page2_infor = ct_infor[page_id + 1]
                        page_img2 = page2_infor[0]
                        paragr1_infor = page1_infor[1][-1]
                        paragr2_infor = page2_infor[1][0]
                        if CheckConditionsToMergePars(page_img1, page_img2, paragr1_infor, paragr2_infor).ntm_2pars:
                            total_paragraphs[page_id][-1], total_paragraphs[page_id + 1][0] = "", pr_list1[-1] + pr_list2[0]
                        else: 
                            pass
            last_id = self._page_count - 1
            if len(total_paragraphs[last_id]) == 0:
                pass
            else:
                if len(total_paragraphs[last_id]) == 1:
                    pass
                else:
                    para_list = total_paragraphs[last_id]
                    for par_id in range(len(para_list) -1):
                        page_infor = ct_infor[last_id]
                        sp_para_infor1 = page_infor[1][par_id]
                        sp_para_infor2 = page_infor[1][par_id + 1]
                        same_img = page_infor[0]
                        if CheckConditionsToMergePars(same_img, same_img, sp_para_infor1, sp_para_infor2).ntm_2pars:
                            total_paragraphs[last_id][par_id], total_paragraphs[last_id][par_id + 1] = "", para_list[par_id] + para_list[par_id + 1]
                        else:
                            pass
        else:
            if len(total_paragraphs[0]) == 0:
                pass
            else:
                if len(total_paragraphs[0]) == 1:
                    pass
                else:
                    para_list = total_paragraphs[0]
                    for par_id in range(len(para_list) -1):
                        page_infor = ct_infor[0]
                        sp_para_infor1 = page_infor[1][par_id]
                        sp_para_infor2 = page_infor[1][par_id + 1]
                        same_img = page_infor[0]
                        if CheckConditionsToMergePars(same_img, same_img, sp_para_infor1, sp_para_infor2).ntm_2pars:
                            total_paragraphs[0][par_id], total_paragraphs[0][par_id + 1] = "", para_list[par_id] + para_list[par_id + 1]
                        else:
                            pass

        id = 0
        for key in list(ct_infor.keys()):
            for paragr in total_paragraphs[key]:
                if paragr != "":
                    paragr_content[id] = paragr
                    id += 1
                else:
                    pass           
        return paragr_content, outlier
