import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import numpy as np
import pandas as pd
import pickle
import gzip
import glob
import os
from skimage.morphology import disk, closing
from skimage.measure import label

class BoneMetastasis:
    """
    Class to handle CT images and associated data for bone metastasis analysis.

    Args:
        image (str): Path to the CT image file.
        lesion (str): Path to the lesion mask file.
        bone (str): Path to the ts mask file.
        pred (str): Path to the prediction file.
    """
    priority_bones = [25, 26, 27, 31, 32, 43, 44, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]
    
    priority_organs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 90, 51]

    previously_used_labels = [0, 1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 21, 22, 51, 90]
    # Spleen, kidney, gallbladder, liver, stomach, pancreas, lung, adrenal gland,

    def __init__(self, image=None, lesion=None, bone=None, pred=None):
        """Initialize with paths to image, lesion, bone, and prediction files."""
        bone_labels = pd.read_csv("./ts_table.csv", dtype={'Label': int, 'Description': str, 'is_bone': bool})
        bone_labels = bone_labels[bone_labels['is_bone'] == True]['Label'].to_list()

        self.path = {'image': image, 'lesion': lesion, 'bone': bone, 'pred': pred, 'pet_img' : image.replace("0.nii.gz","1.nii.gz") if image else None}
        self.sitk_images = {'image': None, 'lesion': None, 'bone': None, 'pred': None}

        self.lesion = self.load('lesion')
        self.ts = self.load('bone')
        self.pred = self.load('pred')

        self.bone_labels = bone_labels
        self.organ_labels = BoneMetastasis.priority_organs
        self.priority_labels = self.bone_labels + self.organ_labels
        # mask out rest so only bones are kept 
        self.id = os.path.basename(self.path['image']).split('.')[0]        
        self.table = {'ground': pd.DataFrame(), 'pred': pd.DataFrame()}
        self.intersections = self.calculate_intersection()

        self.organ_overlap = self.lesion_overlap(self.pred, self.organ_labels)
        self.bone_overlap = self.lesion_overlap(self.pred, self.bone_labels)
    
    def __str__(self):
        out = []
        out.append("Metrics:")
        out.append(str(getattr(self, "metrics", "None")))
        out.append("Bone Overlap:")
        out.append(str(getattr(self, "bone_overlap", "None")))
        out.append("Organ Overlap:")
        out.append(str(getattr(self, "organ_overlap", "None")))
        return "\n".join(out)

    def load(self, image_type):
        """Load a .nii.gz file."""
        if not self.path[image_type]:
            print(f"No path provided for loading {image_type}")
            return None
        try:
            # load image as a SimpleITK image
            self.sitk_images[image_type] = sitk.ReadImage(self.path[image_type])
            return nib.load(self.path[image_type]).get_fdata()
        except Exception as e:
            print(f"Error loading {image_type} from {self.path[image_type]}: {e}")
            return None

    def lesion_overlap(self, mask, labels = None):
        """Calculate overlap metrics between a mask and bone images."""
        if mask is None:
            return None
        
        if labels:
            segments = np.isin(self.ts, labels)
        else: 
            segments = self.ts 

        overlap_mask = (mask > 0) & (segments > 0)
        overlap_count = np.sum(overlap_mask)
        lesion_count = np.sum(mask > 0)
        overlap_percentage = round(overlap_count / lesion_count * 100 if lesion_count > 0 else 0, 2)
        return {
            'overlap_count': overlap_count,
            'lesion_count': lesion_count,
            'segment_count': np.sum(segments > 0),
            'overlap_percentage': overlap_percentage
        }

    def lookup_table(self, table_dir="./ts_table.csv"):
        """Add a lookup table for ts labels."""

        lookup_table = pd.read_csv(table_dir, dtype={'Label': int, 'Description': str, 'is_bone': bool})
        lookup_table = lookup_table[lookup_table['Label'].isin(self.priority_labels)]

        label_counts = {label: np.sum(self.ts == label) for label in self.priority_labels if label > 0}
        lookup_table['Count'] = lookup_table['Label'].map(label_counts).fillna(0).astype(int)
        lookup_table['Type'] = np.select(
            [
                lookup_table['Label'].isin(self.bone_labels),
                lookup_table['Label'].isin(self.organ_labels)
            ],
            [
                'Bone',
                'Organ'
            ],
            default='Other'
        )
        self.table['ground'] = lookup_table
        self.table['pred'] = lookup_table.copy()

    def add_voxel_volume(self):
        """Calculate voxel volume in mm^3."""
        image_path = self.path['lesion'] or self.path['image']
        spacing = sitk.ReadImage(image_path).GetSpacing()
        self.voxel_volume = spacing[0] * spacing[1] * spacing[2]


    def add_priority(self, table_name):
        """Add priority and overlap metrics for a table."""
        # Get Mask
        mask = self.lesion if table_name == 'ground' else self.pred

        # Label which segments are of priority. This is all bones, and the 10 organs
        self.table[table_name]['Priority'] = self.table[table_name]['Label'].isin(BoneMetastasis.priority_bones + BoneMetastasis.priority_organs)

        # Find the overlap of the lesions (ground or pred) with the priority segments
        overlap_counts = {label: np.sum((self.ts == label) & (mask > 0)) for label in self.priority_labels if label > 0}

        # Map overlap counts to the table
        self.table[table_name]['Overlap_Count'] = self.table[table_name]['Label'].map(overlap_counts).fillna(0).astype(int)
        self.table[table_name]['Overlap_Volume'] = self.table[table_name]['Overlap_Count'] * self.voxel_volume
        self.table[table_name]['Overlap_Percentage'] = (
            self.table[table_name]['Overlap_Count'] / self.table[table_name]['Count'] * 100
        ).round(2)

        self.table[table_name].sort_values(by='Priority', ascending=False, inplace=True)

    def calculate_intersection(self):
        """
        Calculates the intersection between the pred lesion and the ground lesion. 
        This ensures that the predicted lesions are accurately compared to the ground truth.

        Returns:
            _type_: _description_
        """
        if self.lesion is None or self.pred is None:
            return None
        intersections = {}

        for label in self.priority_labels:
            if label == 0:
                continue
            mask = (self.ts == label)
            ground_overlap = (self.lesion > 0) & mask
            pred_overlap = (self.pred > 0) & mask
            intersection = np.sum(ground_overlap & pred_overlap)
            intersections[label] = intersection
        return intersections

    def add_high_risk(self):
        """Add high-risk criteria based on overlap and intersection."""
        for table_name in ['ground', 'pred']:
            if table_name not in self.table:
                continue
            
            # nothing is high risk by default
            self.table[table_name]['High Risk'] = False
            
            if table_name == 'ground':
                # For ground truth: priority + overlap OR volume threshold
                self.table[table_name].loc[
                    self.table[table_name]['Priority'] & (self.table[table_name]['Overlap_Percentage'] > 0), 'High Risk'
                ] = True
                self.table[table_name].loc[
                    self.table[table_name]['Overlap_Volume'] >= 4180, 'High Risk'
                ] = True
            else:  # pred
                # For predictions: must have intersection >= 1 AND (priority + overlap OR volume threshold)
                intersection_condition = self.table[table_name]['Label'].map(self.intersections) >= 1
                priority_overlap_condition = (self.table[table_name]['Priority'] & 
                                            (self.table[table_name]['Overlap_Percentage'] > 0))
                volume_condition = self.table[table_name]['Overlap_Volume'] >= 4180
                
                # intersection condition, and priority or overlap condition. 
                self.table[table_name].loc[
                    intersection_condition & (priority_overlap_condition | volume_condition), 'High Risk'
                ] = True

    def merge_tables(self):
        """Merge ground truth and predicted tables."""
        ground_table = self.table['ground']
        pred_table = self.table['pred']
        overlap_risk_cols = ['Overlap_Count', 'Overlap_Percentage', 'Overlap_Volume', 'High Risk']
        cols_to_keep = ['Label'] + [
            col for col in pred_table.columns if col in overlap_risk_cols or col not in ground_table.columns
        ]
        merged_table = ground_table.merge(
            pred_table[cols_to_keep], on='Label', suffixes=('_ground', '_pred'), how='outer'
        )
        if self.intersections:
            merged_table['Intersection'] = merged_table['Label'].map(self.intersections).fillna(0)

        # drop index
        merged_table = merged_table.reset_index(drop=True)
        merged_table.drop(columns=['is_bone'], inplace=True)

        return merged_table

    def calculate_metrics(self): 
        # Convert to boolean arrays for overlap calculation
        lesion_mask = self.lesion > 0
        pred_mask = self.pred > 0
        intersection = lesion_mask & pred_mask
        dice = 2 * intersection.sum() / (lesion_mask.sum() + pred_mask.sum())
        # sensitivity of getting the high risk
        true_positives = self.merged_table['High Risk_ground'] & self.merged_table['High Risk_pred']
        false_negatives = self.merged_table['High Risk_ground'] & ~self.merged_table['High Risk_pred']
        if (true_positives + false_negatives).sum() == 0:
            sensitivity = 0
        else:
            sensitivity = true_positives.sum() / (true_positives.sum() + false_negatives.sum())

        # for only bones
        true_positives_bones = self.merged_table[self.merged_table['Type'] == 'Bone']['High Risk_ground'] & self.merged_table[self.merged_table['Type'] == 'Bone']['High Risk_pred']
        false_negatives_bones = self.merged_table[self.merged_table['Type'] == 'Bone']['High Risk_ground'] & ~self.merged_table[self.merged_table['Type'] == 'Bone']['High Risk_pred']
        if (true_positives_bones + false_negatives_bones).sum() == 0:
            sensitivity_bones = 0
        else:
            sensitivity_bones = true_positives_bones.sum() / (true_positives_bones.sum() + false_negatives_bones.sum())
    
        # for only organs
        true_positives_organs = self.merged_table[self.merged_table['Type'] == 'Organ']['High Risk_ground'] & self.merged_table[self.merged_table['Type'] == 'Organ']['High Risk_pred']
        false_negatives_organs = self.merged_table[self.merged_table['Type'] == 'Organ']['High Risk_ground'] & ~self.merged_table[self.merged_table['Type'] == 'Organ']['High Risk_pred']
        if (true_positives_organs + false_negatives_organs).sum() == 0:
            sensitivity_organs = 0
        else:
            sensitivity_organs = true_positives_organs.sum() / (true_positives_organs.sum() + false_negatives_organs.sum())

        return pd.DataFrame({"file": self.id, "dice": dice, "sensitivity": sensitivity, "sensitivity organs": sensitivity_organs, "sensitivity bones": sensitivity_bones}, index=[0])

##########################
#
#  LESION EVALUATION CLASS
# 
##########################


class LesionEval:
    def __init__(self,id, pred, image, ground):
        self.id = id
        self.image = image
        self.pred = pred
        self.ground = ground
        self.ground_lesion_candidates = self.form_lesion_candidates_from_annotation(self.ground)[0]
        self.pred_lesion_candidates = self.form_lesion_candidates_from_annotation(self.pred)[0]

    def form_lesion_candidates_from_annotation(self, annotation):
        # function to form lesion candidates from annotation

        # Get spacing from SimpleITK image object
        spacing = self.image.GetSpacing() 
        print(f"Spacing: {spacing}")

        anno_np = sitk.GetArrayFromImage(annotation)

        # create a 3D structuring element to help with morphological operations
        margin = 5 / spacing[0]
        strel = disk(int(margin))
        
        margin = 0.5 / spacing[0]
        strel2 = disk(int(margin))
        strel2 = np.pad(strel2, 1, 'constant')

        # Stack 3 structuring elements to make in 3D...
        target_shape = np.maximum(strel.shape, strel2.shape)
        strel = pad_to_shape(strel, target_shape)
        strel2 = pad_to_shape(strel2, target_shape)
        strel_total = np.stack([strel2, strel, strel2])

        # perform morphological closing to fill holes
        closed_annotation_np = closing(anno_np, strel_total)
        print(f"Shape of closed annotation image: {np.shape(closed_annotation_np)}")
        
        # perform connected component analysis to count the number of lesions
        lesions, num_lesions = label(closed_annotation_np, return_num=True)
        print(f"Number of lesions detected: {num_lesions}")
        
        return lesions, num_lesions

    def save_intersection(self, output_path):
        """ Save intesection as a .nii.gz file """
        if self.lesion_candidates is None:
            print("No lesion candidates to save.")
            return
        
        # Make a intersections folder 
        if not os.path.exists('intersections'):
            os.makedirs('intersections')

        output_path = os.path.join('intersections', output_path)
        
        # Extract lesion array from tuple (lesion_candidates returns a tuple)
        lesion_array = self.lesion_candidates[0] if isinstance(self.lesion_candidates, tuple) else self.lesion_candidates
        
        # Convert lesion candidates to SimpleITK image
        sitk_image = sitk.GetImageFromArray(lesion_array)
        sitk_image.CopyInformation(self.ground)

        # Save the image
        sitk.WriteImage(sitk_image, output_path)
        print(f"Intersection saved to {output_path}")

def pad_to_shape(arr, target_shape):
    pad_y = (target_shape[0] - arr.shape[0]) // 2
    pad_x = (target_shape[1] - arr.shape[1]) // 2
    return np.pad(arr, ((pad_y, target_shape[0] - arr.shape[0] - pad_y),
                        (pad_x, target_shape[1] - arr.shape[1] - pad_x)),mode='constant')

def resample_to_match(src_image, target_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(src_image)