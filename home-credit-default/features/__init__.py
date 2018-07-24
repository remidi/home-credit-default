import os
import pandas as pd
from sklearn.externals import joblib

from attrdict import AttrDict


class FeatureBuilder(object):
    """ Abstract class for building features """

    def __init__(self, config):
        self.config = AttrDict(config)
        self.csv_path = None
        self.version_path = None
        self.latest_version = None

    def _feature_joiner(self, merge_col, feats):
        new_version = pd.merge(left=self.latest_version, right=feats, on=merge_col, how='left')
        return new_version

    @staticmethod
    def _feature_naming(original_feats=list(), addition=list()):
        feat_cols = ['{}_{}'.format(col, agg) for col, agg in zip(original_feats, addition)]
        return feat_cols

    def _check_features(self, feats):
        self.latest_version = joblib.load(self.version_path)
        for feat in feats.columns:
            assert feat not in self.latest_version.columns, \
                'Column : {} already exists in: {}'.format(feat,
                                                           self.version_path.split('/')[-1])

    def persist_verison(self, feats, merge_col):
        self._check_features(feats)
        save_version = self._feature_joiner(merge_col, feats)
        joblib.dump(save_version, self.version_path)

    def groupby(self, feats, aggregations, *args, **kwargs):
        pass
