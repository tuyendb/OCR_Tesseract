import os 
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from utils.util import *


class GroupContent:
    
    def __init__(self, paragraphs_infor, group):
        self._paragraphs_infor = paragraphs_infor
        self._group = group
        self._para_id_list = group['para_ids']
    
    @property
    def words_count(self):
        words_num = 0
        for para_id in self._para_id_list:
            for line in self._paragraphs_infor[para_id][0]:
                words_num += len(line)
        return words_num

    @property
    def lines_count(self):
        lines_num = 0
        for para_id in self._para_id_list:
            lines_num += len(self._paragraphs_infor[para_id][0])
        return lines_num
        