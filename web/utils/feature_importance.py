import pickle
import numpy as np
import matplotlib.pyplot as plt

feature_names = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'age',
       'area_percentage', 'height_percentage', 'has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other',
       'count_families', 'has_secondary_use_agriculture',
       'has_secondary_use_hotel', 'has_secondary_use_rental',
       'has_secondary_use_institution', 'has_secondary_use_school',
       'has_secondary_use_industry', 'has_secondary_use_health_post',
       'has_secondary_use_gov_office', 'has_secondary_use_use_police',
       'has_secondary_use_other', 'land_surface_condition_n',
       'land_surface_condition_o', 'land_surface_condition_t',
       'foundation_type_h', 'foundation_type_i', 'foundation_type_r',
       'foundation_type_u', 'foundation_type_w', 'roof_type_n', 'roof_type_q',
       'roof_type_x', 'ground_floor_type_f', 'ground_floor_type_m',
       'ground_floor_type_v', 'ground_floor_type_x', 'ground_floor_type_z',
       'other_floor_type_j', 'other_floor_type_q', 'other_floor_type_s',
       'other_floor_type_x', 'position_j', 'position_o', 'position_s',
       'position_t', 'plan_configuration_a', 'plan_configuration_c',
       'plan_configuration_d', 'plan_configuration_f', 'plan_configuration_m',
       'plan_configuration_n', 'plan_configuration_o', 'plan_configuration_q',
       'plan_configuration_s', 'plan_configuration_u',
       'legal_ownership_status_a', 'legal_ownership_status_r',
       'legal_ownership_status_v', 'legal_ownership_status_w']

def load_model(algorithm_name):
       if algorithm_name  == 'softmax':
              with open('models/softmax_model.pickle', 'rb') as f:
                     model = pickle.load(f)
                     return model
       elif algorithm_name == 'decisiontree':
              with open('models/decision_tree_model.pickle', 'rb') as f:
                     model = pickle.load(f)
                     return model


def make_predictions(algorithm_name, feature_values):
       model = load_model(algorithm_name)
       features = np.array(feature_values).reshape(1,-1)
       predicted_grade = model.predict(features)
       return predicted_grade


def save_importance(algorithm_name):
       model = load_model(algorithm_name)
       if algorithm_name == 'softmax':
              importances = np.abs(model.coef_).sum(axis=0)
              # Top 10 importances
              importances = importances[:10]

              indices = np.argsort(importances)[::-1]
              important_feature = []
              for i in indices:
                     important_feature.append(feature_names[i])
              plt.bar(range(10), importances[indices])
              plt.xticks(range(10), important_feature, rotation=90)
              plt.savefig('static/images/softmax_imp.png', dpi=200)
              filename = 'static/images/softmax_imp.png'
              return filename
       elif algorithm_name == 'decisiontree':
              # Get the feature importances
              importance = model.feature_importances_

              importance = importance[:10]

              sorted_idx = importance.argsort()[::-1]

              important_feature = []
              for i in sorted_idx:
                     important_feature.append(feature_names[i])

              # Create a bar plot of the feature importances
              plt.bar(range(10), importance[sorted_idx])
              plt.xticks(range(10), important_feature , rotation=90)
              plt.savefig('static/images/softmax_imp.png', dpi=200)
              filename = 'static/images/softmax_imp.png'
              return filename
       else:
              pass

      

