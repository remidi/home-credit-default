import os
import pandas as pd
from sklearn.externals import joblib

from attrdict import AttrDict

import sys
sys.path.append('../')


class FeatureBuilder(object):
    """ Abstract class for building features """

    def __init__(self):
        self.data_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2] + ['input_data'])
        self.feat_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2] + ['features'])
        self.csv_path = None
        self.version_path = None
        self.latest_version = None
        self.raw_data = None
        self.features = None

    @staticmethod
    def feature_save(func):
        def wrapper(merge_col):
            feats = func()
            self.features = pd.merge(self.features, feats, on=merge_col, how='left')
            return wrapper

    def _feature_joiner(self, merge_col, feats):
        new_version = pd.merge(left=self.latest_version, right=feats, on=merge_col, how='left')
        return new_version

    def load_raw(self):
        self.raw_data = pd.read_csv(self.csv_path)

    @staticmethod
    def aggregate_feature_naming(feat_cols=list(), task=list()):
        feat_cols = ['{}_{}'.format(col, agg) for col, agg in zip(feat_cols, task)]
        return feat_cols

    @staticmethod
    def interact_feature_naming(feat_cols, task):
        pass

    def load_latest(self):
        self.latest_version = joblib.load(self.version_path)
        return self.latest_version

    def _check_features(self, feats):
        self.load_latest()
        for feat in feats.columns:
            assert feat not in self.latest_version.columns, \
                'Column : {} already exists in: {}'.format(feat,
                                                           self.version_path.split('/')[-1])

    def persist(self, feats, merge_col):
        self._check_features(feats)
        save_version = self._feature_joiner(merge_col, feats)
        joblib.dump(save_version, self.version_path)

    def aggregations(self, feats, aggregates, *args, **kwargs):
        pass

    def interactions(self, feat_tuples, interacts, *args, **kwargs):
        pass

    @staticmethod
    def _hand_crafted_features(app_data):
        app_data['annuity_income_percentage'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']
        app_data['car_to_birth_ratio'] = app_data['OWN_CAR_AGE'] / app_data['DAYS_BIRTH']
        app_data['car_to_employ_ratio'] = app_data['OWN_CAR_AGE'] / app_data['DAYS_EMPLOYED']
        app_data['children_ratio'] = app_data['CNT_CHILDREN'] / app_data['CNT_FAM_MEMBERS']
        app_data['credit_to_annuity_ratio'] = app_data['AMT_CREDIT'] / app_data['AMT_ANNUITY']
        app_data['credit_to_goods_ratio'] = app_data['AMT_CREDIT'] / app_data['AMT_GOODS_PRICE']
        app_data['credit_to_income_ratio'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']
        app_data['days_employed_percentage'] = app_data['DAYS_EMPLOYED'] / app_data['DAYS_BIRTH']
        app_data['income_credit_percentage'] = app_data['AMT_INCOME_TOTAL'] / app_data['AMT_CREDIT']
        app_data['income_per_child'] = app_data['AMT_INCOME_TOTAL'] / (1 + app_data['CNT_CHILDREN'])
        app_data['income_per_person'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS']
        app_data['payment_rate'] = app_data['AMT_ANNUITY'] / app_data['AMT_CREDIT']
        app_data['phone_to_birth_ratio'] = app_data['DAYS_LAST_PHONE_CHANGE'] / app_data['DAYS_BIRTH']
        app_data['phone_to_employ_ratio'] = app_data['DAYS_LAST_PHONE_CHANGE'] / app_data['DAYS_EMPLOYED']

        app_data['cnt_non_child'] = app_data['CNT_FAM_MEMBERS'] - app_data['CNT_CHILDREN']
        app_data['child_to_non_child_ratio'] = app_data['CNT_CHILDREN'] / app_data['cnt_non_child']
        app_data['income_per_non_child'] = app_data['AMT_INCOME_TOTAL'] / app_data['cnt_non_child']
        app_data['credit_per_person'] = app_data['AMT_CREDIT'] / app_data['CNT_FAM_MEMBERS']
        app_data['credit_per_child'] = app_data['AMT_CREDIT'] / (1 + app_data['CNT_CHILDREN'])
        app_data['credit_per_non_child'] = app_data['AMT_CREDIT'] / app_data['cnt_non_child']

        return app_data

    @staticmethod
    def _groupby_features(app_data):

        groupby_aggregate_names = []
        for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
            group_object = app_data.groupby(groupby_cols)
            for select, agg in tqdm(specs):
                groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
                app_data = app_data.merge(group_object[select]
                            .agg(agg)
                            .reset_index()
                            .rename(index=str,
                                    columns={select: groupby_aggregate_name})
                            [groupby_cols + [groupby_aggregate_name]],
                            on=groupby_cols,
                            how='left')
                groupby_aggregate_names.append(groupby_aggregate_name)


        return app_data


    @staticmethod
    def _groupby_diffs(app_data):
        app_data = 0
        diff_feature_names = []
        for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
            for select, agg in tqdm(specs):
                if agg in ['mean', 'median', 'max', 'min']:
                    groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
                    diff_name = '{}_diff'.format(groupby_aggregate_name)
                    abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)

                    app_data[diff_name] = app_data[select] - app_data[groupby_aggregate_name]
                    app_data[abs_diff_name] = np.abs(app_data[select] - app_data[groupby_aggregate_name])

                    diff_feature_names.append(diff_name)
                    diff_feature_names.append(abs_diff_name)

        return app_data, diff_feature_names

