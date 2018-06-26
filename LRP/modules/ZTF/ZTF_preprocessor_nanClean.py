#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 23:59:25 2018

preprocessor of ZTF data that includes NaN cleaning

@author: asceta
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from modules.ZTF.ZTF_preprocessor import ZTF_data_preprocessor

class ZTF_data_preprocessor_clean_nans(ZTF_data_preprocessor):
    
    """
    Cosntructor
    """
    def __init__(self, data_path = None):
        super().__init__(data_path)
    
    #Override
    def get_preprocessed_data(self, path = None):
        if path is None:
            path = self.data_path
        
        list_samples = self.json2list(path)
        list_sample_NoNan = self.clean_nans(list_samples)
        numpy_misshape_clean_samples = self.clean_misshaped(list_sample_NoNan)
        cropped_at_center_samples = self.crop_at_center(numpy_misshape_clean_samples)
        #zero_filled_samples = self.zero_fill_nans(cropped_at_center_samples)
        normalized_samples = self.normalize_01(cropped_at_center_samples)#zero_filled_samples)
        
        result = normalized_samples
        return result