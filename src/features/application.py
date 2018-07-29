import pandas as pd
import numpy as np

import os
import sys

from __init__ import FeatureBuilder

import utils


class ApplicationBuilder(FeatureBuilder):
    def __init__(self):
        super(ApplicationBuilder, self).__init__()
        self.csv_path = self.data_dir + '/application_train.csv'
        self.version_path = self.feat_dir + '/application.pkl'

    @FeatureBuilder.feature_save('SK_CURR_ID')
    def base_cols(self):
        pass


if __name__ == '__main__':
    a = ApplicationBuilder()
    a.load_raw()

