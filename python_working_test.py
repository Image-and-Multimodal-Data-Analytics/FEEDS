# %%
import os 
from pathlib import Path



# %%
os.chdir('//dartfs/rc/lab/B/BhattacharyaI/Results/nnUNet_data/')
UNCERTAINTY_MAPS_DIR = './nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_10_ens_comb/validation_uncertainty_maps/'
PREDICTED_LABELS_DIR = './nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_10_ens_comb/validation_predictions/'
SUMMARY_JSON_DIR = './nnUNet_results/Dataset999_AutoPet/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres/fold_10_ens_comb/validation_predictions/summary.json'
TRUE_LABELS_DIR = '//dartfs/rc/lab/B/BhattacharyaI/Public_Datasets/Autopet_III_nnunet_raw/Dataset888_AutoPet/labelsTr'
SAVE_DIR = '//dartfs/rc/lab/B/BhattacharyaI/Results/Biratal/Uncertainty_Examples/'
    

print(os.path.exists(UNCERTAINTY_MAPS_DIR))
print(os.path.exists(PREDICTED_LABELS_DIR))
# print(os.path.exists(SUMMARY_JSON_DIR))
print(os.path.exists(TRUE_LABELS_DIR))

print(os.path.exists(SAVE_DIR))

# load a random label file from my directory and the correct directory and compare them
import numpy as np
import json 

with open(SUMMARY_JSON_DIR, 'r') as f:
    summary_json = json.load(f)
per_cases = summary_json['metric_per_case']
  
label_prefix = ""
X = "_total_uncertainty"
files_with_prefix = [f for f in os.listdir(UNCERTAINTY_MAPS_DIR) if f.startswith(label_prefix) and f.endswith(f'{X}.nii.gz')]
file_names_in_true = [f.replace(f'{X}', '').strip() for f in files_with_prefix]
file_dice = []

for case in per_cases:
    if file_names_in_true[0][:10] in case['prediction_file']:
        file_dice.append(case['metrics']['1']['Dice'])
        break
    
# print(files_with_prefix)
# print(file_names_in_true)

# %%
import os

# Debug step by step


# %%
files_with_prefix[0]

# %%
import nibabel as nib
uncertainty_map_path = os.path.join(UNCERTAINTY_MAPS_DIR, files_with_prefix[0])
uncertainty_map = nib.load(os.path.abspath(uncertainty_map_path)).get_fdata()

predicted_label_path = os.path.join(PREDICTED_LABELS_DIR, files_with_prefix[0].replace(f'{X}', ''))
predicted_label = nib.load(predicted_label_path).get_fdata()
print(uncertainty_map.shape)

true_label_path = os.path.join(TRUE_LABELS_DIR, file_names_in_true[0])
print(true_label_path)
print(os.path.exists(true_label_path))
true_label = nib.load(true_label_path).get_fdata()
print(true_label.shape)


# %%
# print value_counts of the true label
unique, counts = np.unique(true_label, return_counts=True)
print(unique)
print(counts)

# %%
# Check the uncertainty of the predicted labels
# # positions of the predicted labels
# np.where(predicted_label == 1)
# THRESHOLD = 0.001

# uncertainty_at_predicted = uncertainty_map[np.where(predicted_label == 1)]
# uncertainty_above_0 = uncertainty_map[np.where(uncertainty_map > THRESHOLD)]

# mean_uncertainty_at_predicted = np.mean(uncertainty_at_predicted)
# mean_uncertainty_thresholded  = np.mean(uncertainty_above_0)

# print(len(uncertainty_at_predicted), mean_uncertainty_at_predicted)
# print(len(uncertainty_above_0), mean_uncertainty_thresholded)

# # %%
# # plot a histogram of the uncertainty_map_values
# import matplotlib.pyplot as plt
# plt.hist(uncertainty_at_predicted, bins=50)
# plt.xlabel('Uncertainty')
# plt.ylabel('Frequency')

# # %%
# # show heatmap of uncertainty map
# import matplotlib.pyplot as plt

# # Use MEAN instead of MAX for uncertainty (more representative)
# z_project_unc = np.mean(uncertainty_map, axis=2)  # changed from np.max
# z_project = np.max(true_label, axis=2)
# z_project_pred = np.max(predicted_label, axis=2)

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# ax[0].imshow(z_project, cmap='gray')
# ax[0].set_title('Z-projection of True Label')

# ax[1].imshow(z_project_pred, cmap='gray')
# ax[1].set_title('Z-projection of Predicted Label')

# im_unc = ax[2].imshow(
#     z_project_unc,
#     cmap='hot',
#     interpolation='nearest',
#     vmin=0,
#     vmax=np.percentile(z_project_unc, 99)  # ignore extreme outliers
# )
# ax[2].set_title('Z-projection of Uncertainty Map (Mean)')
# fig.colorbar(im_unc, ax=ax[2], fraction=0.046, pad=0.04, label='Uncertainty Value')

# for axis in ax:
#     axis.set_xlabel('X-axis')
#     axis.set_ylabel('Y-axis')

# case_name = file_names_in_true[0].replace('.nii.gz', '')
# plt.suptitle(
#     f"{case_name}\nDice: {np.round(file_dice, 4)}      Mean Uncertainty of Predicted Voxels: {np.round(np.mean(mean_uncertainty_at_predicted), 4)}"
# )
# plt.tight_layout()
# plt.show()

# # %%
# z_project_unc = np.max(uncertainty_map, axis=2)
# plt.imshow(z_project_unc)

# # %%
# import os
# import json
# from pathlib import Path

# import nibabel as nib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# rng = np.random.default_rng(42)


# def extract_case_metrics(case):
#     metrics = case.get("metrics", {})
#     dice_value = np.nan

#     if isinstance(metrics, dict):
#         if "1" in metrics and isinstance(metrics["1"], dict) and "Dice" in metrics["1"]:
#             dice_value = metrics["1"]["Dice"]
#         else:
#             for value in metrics.values():
#                 if isinstance(value, dict) and "Dice" in value:
#                     dice_value = value["Dice"]
#                     break

#     return {
#         "dice": float(dice_value) if np.isfinite(dice_value) else np.nan,
#     }


# def get_case_record(case):
#     prediction_name = Path(case.get("prediction_file", "")).name
#     if not prediction_name:
#         return None

#     case_id = prediction_name.replace(".nii.gz", "")
#     uncertainty_name = f"{case_id}_total_uncertainty.nii.gz"

#     uncertainty_path = Path(UNCERTAINTY_MAPS_DIR) / uncertainty_name
#     predicted_path = Path(PREDICTED_LABELS_DIR) / prediction_name
#     true_path = Path(TRUE_LABELS_DIR) / prediction_name

#     if not uncertainty_path.exists() or not predicted_path.exists():
#         return None

#     uncertainty_map = nib.load(str(uncertainty_path)).get_fdata()
#     predicted_label = nib.load(str(predicted_path)).get_fdata()

#     predicted_mask = predicted_label > 0
#     uncertainty_pred = uncertainty_map[predicted_mask]

#     has_ground_truth = true_path.exists()
#     gt_positive_fraction = np.nan
    
#     if has_ground_truth:
#         true_label = nib.load(str(true_path)).get_fdata()
#         gt_positive_fraction = float(np.mean(true_label >= 1))

#     metrics = extract_case_metrics(case)

#     return {
#         "case_id": case_id,
#         "prediction_file": prediction_name,
#         "uncertainty_file": uncertainty_name,
#         "has_ground_truth": bool(has_ground_truth),
#         "dice": metrics["dice"] if has_ground_truth else np.nan,
#         "mean_uncertainty_predicted_voxels": float(np.mean(uncertainty_pred)) if uncertainty_pred.size else np.nan,
#         "median_uncertainty_predicted_voxels": float(np.median(uncertainty_pred)) if uncertainty_pred.size else np.nan,
#         "mean_uncertainty_image": float(np.mean(uncertainty_map)),
#         "std_uncertainty_image": float(np.std(uncertainty_map)),
#         "predicted_voxel_count": int(np.count_nonzero(predicted_mask)),
#         "predicted_voxel_fraction": float(np.mean(predicted_mask)),
#         "uncertainty_fraction_above_0.001": float(np.mean(uncertainty_map > 0.001)),
#         "gt_positive_fraction": gt_positive_fraction,
#     }

# # %%
# import tqdm
# with open(SUMMARY_JSON_DIR, "r") as f:
#     summary_json = json.load(f)

# per_cases = summary_json.get("metric_per_case", [])

# valid_case_records = []
# for case in tqdm.tqdm(per_cases):
#     rec = get_case_record(case)
#     if rec is not None:
#         valid_case_records.append(rec)
#     else: 
#         print(f"Warning: Case {case.get('prediction_file', 'unknown')} is missing required files and will be skipped.")



# # %%
# # convert list of dicts to dataframe and sort by has_ground_truth and dice
# # make df long not wide

# list_of_dfs = []
# for rec in valid_case_records:
#     temp_df = pd.DataFrame([rec])
#     list_of_dfs.append(temp_df)
    
# big_df = pd.concat(list_of_dfs, ignore_index=True)


# # %%
# # count how many are 0 

# df_without_gt = big_df.loc[big_df['gt_positive_fraction'] == 0]
# df_with_gt = big_df[big_df["gt_positive_fraction"] > 0].copy()
# # set had_ground_truth to False for df_without_gt and True for df_with_gt
# df_without_gt.loc[df_without_gt.index, 'has_ground_truth'] = False
# df_with_gt.loc[df_with_gt.index, 'has_ground_truth'] = True

# # %%
# fig, axes = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

# DF_PLOT = df_with_gt


# s1 = axes.scatter(
#     DF_PLOT["mean_uncertainty_predicted_voxels"],
#     DF_PLOT["dice"],
#     alpha=0.9,
#     s=40,
# )
# axes.set_title("Dice vs uncertainty in predicted voxels")
# axes.set_xlabel("Mean uncertainty in predicted voxels")
# axes.set_ylabel("Dice")

# x = axes.collections[0].get_offsets()[:, 0]
# y = axes.collections[0].get_offsets()[:, 1]
# valid = np.isfinite(x) & np.isfinite(y)
# if np.count_nonzero(valid) >= 2 and np.unique(x[valid]).size > 1:
#     slope, intercept = np.polyfit(x[valid], y[valid], 1)
#     xx = np.linspace(x[valid].min(), x[valid].max(), 100)
#     axes.plot(xx, slope * xx + intercept, color="crimson", linewidth=2)

# plt.suptitle(f"Uncertainty vs Dice across all validation cases ({len(DF_PLOT)})")
# plt.show()



# # %%
# DF_PLOT.columns

# # %%
# fig, axes = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

# DF_PLOT = df_without_gt

# # do a histogram of the mean uncertainty in predicted voxels for cases without ground truth
# import seaborn as sns
# sns.histplot(df_without_gt["mean_uncertainty_predicted_voxels"], bins=20, ax=axes, label = f"Without GT (n = {len(df_without_gt)})", color="blue", alpha=0.5)
# sns.histplot(df_with_gt["mean_uncertainty_predicted_voxels"], bins=20, ax=axes, label = f"With GT (n = {len(df_with_gt)})", color="orange", alpha=0.5)
# axes.set_title("Histogram of mean uncertainty in predicted voxels for cases without ground truth\n Validation set")
# axes.legend()



# # %%
# np.sum(df_without_gt['mean_uncertainty_predicted_voxels'] > 0.8)

# # %%
# # with the gt have a histogram of the dice
# fig, axes = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
# sns.histplot(df_with_gt["dice"], bins=20, ax=axes, label = f"With GT (n = {len(df_with_gt)})", color="orange", alpha=0.5)
# axes.set_title(f"Histogram of Dice scores for cases with ground truth\n Validation set (n = {len(df_with_gt)})")


# # %%
# fig, axes = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)


# box_data = []
# labels = []
# if len(df_with_gt) > 0:
#     box_data.append(df_with_gt["mean_uncertainty_predicted_voxels"].dropna())
#     labels.append(f"With GT ({len(df_with_gt)})")
# box_data.append(df_without_gt["mean_uncertainty_predicted_voxels"].dropna())
# labels.append(f"Without GT ({len(df_without_gt)})")

# axes.boxplot(box_data, tick_labels=labels, patch_artist=True, boxprops={"facecolor": "#dce7f5"})
# axes.set_title("Whole-image uncertainty distribution")
# axes.set_ylabel("Mean uncertainty")

# plt.show()


# # %%
# feature_columns = [
#     "mean_uncertainty_predicted_voxels",
#     "mean_uncertainty_image",
#     "predicted_voxel_fraction",
#     "predicted_voxel_count",
#     "std_uncertainty_image",
# ]

# regression_df = df_with_gt.dropna(subset=feature_columns + ["dice"]).copy()

# if len(regression_df) >= 5:
#     X = regression_df[feature_columns].to_numpy(dtype=float)
#     y = regression_df["dice"].to_numpy(dtype=float)

#     X_mean = X.mean(axis=0)
#     X_std = X.std(axis=0)
#     X_std[X_std == 0] = 1.0
#     Xz = (X - X_mean) / X_std

#     design = np.c_[np.ones(len(Xz)), Xz]
#     coefs = np.linalg.lstsq(design, y, rcond=None)[0]
#     pred = design @ coefs

#     tss = float(np.sum((y - np.mean(y)) ** 2))
#     rss = float(np.sum((y - pred) ** 2))
#     r2 = 1 - rss / tss if tss > 0 else np.nan

#     coef_df = pd.DataFrame({
#         "feature": ["intercept"] + feature_columns,
#         "standardized_coefficient": coefs,
#     }).sort_values("standardized_coefficient", key=lambda s: s.abs(), ascending=False)

#     display(coef_df)
#     print(f"Linear regression R^2: {r2:.3f}")
# else:
#     print("Not enough GT cases for regression (need at least 5).")

# # %%
# summary_cols = [
#     "mean_uncertainty_predicted_voxels",
#     "mean_uncertainty_image",
#     "predicted_voxel_fraction",
#     "predicted_voxel_count",
# ]

# print("Summary by has_ground_truth")
# display(df.groupby("has_ground_truth")[summary_cols].agg(["count", "mean", "std"]))

# print("Top uncertain GT cases")
# if len(df_with_gt) > 0:
#     display(df_with_gt.sort_values("mean_uncertainty_predicted_voxels", ascending=False).head(10))
# else:
#     print("None")

# print("Top uncertain no-GT cases")
# if len(df_without_gt) > 0:
#     display(df_without_gt.sort_values("mean_uncertainty_predicted_voxels", ascending=False).head(10))
# else:
#     print("None")

# # Optional: persist table for downstream analysis
# # df.to_csv(os.path.join(SAVE_DIR, "uncertainty_case_summary_100.csv"), index=False)


