# this file combines all the features into a finalized table for training data

import pandas as pd
import numpy as np

class CombineFeaturesTable():
    """Combine all required features for training data"""
    def __init__(self, initial_features_table, win_rate_type_location_table, coach_stats_table):
        super(CombineFeaturesTable, self).__init__()
        self.win_rate_type_location_table = win_rate_type_location_table
        self.coach_stats_table = coach_stats_table
        self.initial_features_table = initial_features_table

        self.combine_feature_table()
        self.min_max_standardization()
        self.final_input_training_data()

    def combine_feature_table(self):