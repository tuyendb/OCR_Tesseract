import time
import os 
import json
import sys

from matplotlib.font_manager import json_dump
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from utils.util import *
from path.path import *
from objects.book import Book
from objects.page import Page
from objects.text_block import TextBlock
from objects.data_extraction import DataExtraction
from objects.data_request import GetDataFromPDF


def main():
    st_time = time.time()
    pages = Book(pdf_path, None, None).pages
    title_mark =  []
    for page_id in range(pages.__len__()):
        pages[page_id], titles = Page(pages[page_id]).edited_page
        for title in titles:
            title_mark.append([page_id, title])
    print(title_mark)
    extract_data = {}
    for tt_id in range(0, len(title_mark) - 1):
        title_content = TextBlock(title_mark[tt_id][1]).block_content
        extract_data[title_content] = {}
        title1 = title_mark[tt_id]
        title2 = title_mark[tt_id + 1]
        ms_crop_imgs = crop_imgs_between_2_titles(pages, title1, title2)
        mesial_content = DataExtraction(ms_crop_imgs).extracted_content
        extract_data[title_content] = {
            'content': mesial_content[0],
            'outlier': mesial_content[1]
        }
    final_title = title_mark[-1]
    final_crop_imgs = crop_imgs_final(pages, final_title)
    fn_tt_content = TextBlock(final_title[1]).block_content
    final_content = DataExtraction(final_crop_imgs).extracted_content
    extract_data[fn_tt_content] = {
        'content': final_content[0],
        'outlier': final_content[1]
    }
    # total_time = time.time() - st_time
    # print(total_time)
    print(extract_data)



class ExtractDataFromPDF:

    def __init__(self):
        self.Book = Book
        self.Page = Page
        self.TextBlock = TextBlock
        self.DataExtraction = DataExtraction

    def extract_dict(self, pdf_file, first_page, last_page):
        pages = self.Book(pdf_file, first_page, last_page).pages
        title_mark =  []
        for page_id in range(pages.__len__()):
            pages[page_id] = cv2.resize(pages[page_id], (1600, 2200))
            pages[page_id], titles = self.Page(pages[page_id]).edited_page
            for title in titles:
                title_mark.append([page_id, title])
        extract_data = {}
        if len(title_mark) != 0:
            for tt_id in range(0, len(title_mark) - 1):
                title_content = self.TextBlock(title_mark[tt_id][1]).block_content
                extract_data[title_content] = {}
                title1 = title_mark[tt_id]
                title2 = title_mark[tt_id + 1]
                ms_crop_imgs = crop_imgs_between_2_titles(pages, title1, title2)
                mesial_content = self.DataExtraction(ms_crop_imgs).extracted_content
                extract_data[title_content] = {
                    'content': mesial_content[0],
                    'outlier': mesial_content[1]
                }
            final_title = title_mark[-1]
            final_crop_imgs = crop_imgs_final(pages, final_title)
            fn_tt_content = self.TextBlock(final_title[1]).block_content
            final_content = self.DataExtraction(final_crop_imgs).extracted_content
            extract_data[fn_tt_content] = {
                'content': final_content[0],
                'outlier': final_content[1]
            }
        else:
            pass
        return extract_data

    def __call__(self, data: GetDataFromPDF) -> GetDataFromPDF:
        data_extract = self.extract_dict(data.pdf_file, data.first_page, data.last_page)
        data.extracted_data = data_extract
        return data


if __name__ == '__main__':
    main()