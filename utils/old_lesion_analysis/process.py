import os
import glob
import numpy as np
import SimpleITK as sitk

from files import *
from uncertainty import *
from data_structures import BoneMetastasis, LesionEval 
from analysis import *



def process_image(lesion_file, dirs, ONLY_LESION_STATUS):
    """Processes a single image 

    Args:
        lesion_file (_type_): _description_
        image_files (_type_): _description_
        dirs (_type_): _description_
        existing_summary (_type_): _description_
        ONLY_LESION_STATUS (_type_): _description_

    Returns:
        _type_: _description_
    """

    image_id = os.path.basename(lesion_file).split('.')[0]
    print(f"Looking for matching files for lesion ID: {image_id}", flush=True)
    
    # Find matching files using our improved functions
    image_file = find_matching_file(image_id, dirs['images'], "*.nii.gz")
    bone_file = find_matching_file(image_id, dirs['bone'], "*ts.nii.gz")
    pred_file = find_matching_file(image_id, dirs['pred'], "*.nii.gz")
    image_files = {}
    # Check if all required files are found
    if image_file and bone_file and pred_file:
        print(f"Found all matching files for {image_id}:", flush=True)

        print(f"  Image: {os.path.basename(image_file)}", flush=True)
        print(f"  Lesion: {os.path.basename(lesion_file)}", flush=True)
        print(f"  Bone: {os.path.basename(bone_file)}", flush=True)
        print(f"  Pred: {os.path.basename(pred_file)}", flush=True)

        image_files = {
            'image': image_file,
            'lesion': lesion_file,
            'bone': bone_file,
            'pred': pred_file,
        }
    # if image files does not exist, print a warning
    if len(image_files) == 0:
        print(f"Warning: No matching image file found for lesion ID: {image_id}", flush=True)
        print(f"  Searched in: {dirs['images']}", flush=True)
        print(f"  Searched in: {dirs['labels']}", flush=True)
        print(f"  Searched in: {dirs['bone']}", flush=True)
        print(f"  Searched in: {dirs['pred']}", flush=True)
        print(f"  Pattern: *.nii.gz", flush=True)
        
    ref_image = sitk.ReadImage(image_files['image'])
    ref_shape = sitk.GetArrayFromImage(ref_image).shape
    
    # Handle uncertainty data safely
    uncertainty_data = None
    if dirs.get('uncertainty'):  # Check if uncertainty directory exists
        try:
            uncertainty_data = load_uncertainty_maps(image_id, dirs['uncertainty'])
            if uncertainty_data is not None:
                uncertainty_data = align_uncertainty_dimensions(uncertainty_data, ref_shape)
        except Exception as e:
            print(f"Warning: Could not load uncertainty data for {image_id}: {e}")
            uncertainty_data = None
    
    # Handle probability data safely
    probability = None
    if dirs.get('pred'):  # Check if pred directory exists
        try:
            prob_file = find_matching_file(image_id, dirs['pred'], "*.npz")
            if prob_file:  # Check if probability file was found
                probability_data = np.load(prob_file)
                key = list(probability_data.keys())[0]
                probability = probability_data[key]
                
                # Validate probability shape
                if probability.shape != (2,) + ref_shape:
                    print(f"Warning: Probability shape {probability.shape} doesn't match expected {(2,) + ref_shape}")
                    # Try to reshape or handle mismatch
                    if probability.size == 2 * np.prod(ref_shape):
                        probability = probability.reshape((2,) + ref_shape)
                    else:
                        print(f"Cannot reshape probability data, setting to None")
                        probability = None
            else:
                print(f"No probability file found for {image_id}")
        except Exception as e:
            print(f"Warning: Could not load probability data for {image_id}: {e}")
            probability = None
    
    # This gets you the lesions    
    lesion_evaluated = LesionEval(
        id=image_id,
        pred=sitk.ReadImage(image_files['pred']),
        image=ref_image,
        ground=sitk.ReadImage(image_files['lesion'])
    )

    # Now analyze them - pass None values safely
    pred_stats, high_uncertainty_lesions = analyze_predicted_lesions(
        pred_pos=lesion_evaluated.pred_lesion_candidates,
        ground_pos=lesion_evaluated.ground_lesion_candidates, 
        probs=probability,  # Can be None
        uncertainty_data=uncertainty_data,  # Can be None
        id=image_id, 
        ref_image=ref_image
    )

    ground_stats = analyze_ground_truth_lesions(
        ground_pos=lesion_evaluated.ground_lesion_candidates,
        pred_pos=lesion_evaluated.pred_lesion_candidates,
        probs=probability,  # Can be None
        uncertainty_data=uncertainty_data,  # Can be None
        id=image_id, 
        ref_image=ref_image, 
        high_uncertainty_lesions=high_uncertainty_lesions
    )

    image_stats = analyze_image(
        lesion_evaluated.ground_lesion_candidates,
        lesion_evaluated.pred_lesion_candidates,
        probability,  # Can be None
        uncertainty_data,  # Can be None
        image_id, 
        ref_image, 
        high_uncertainty_lesions
    )
    
    return pred_stats, ground_stats, image_stats


def process_bone_mets(lesion_file, dirs):
    """Processes bone metastasis images

    Args:
        lesion_file (_type_): _description_
        image_files (_type_): _description_
        dirs (_type_): _description_
        existing_summary (_type_): _description_
        ONLY_LESION_STATUS (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Implement the processing logic for bone metastasis images
    image_id = os.path.basename(lesion_file).split('.')[0]
    print(f"Looking for matching files for lesion ID: {image_id}", flush=True)
    
    # Find matching files using our improved functions
    image_file = find_matching_file(image_id, dirs['images'], "*0.nii.gz")
    pet_image_file = find_matching_file(image_id, dirs['images'], "*1.nii.gz")
    bone_file = find_matching_file(image_id, dirs['bone'], "*ts.nii.gz")
    pred_file = find_matching_file(image_id, dirs['pred'], "*.nii.gz")
    print(image_file)
    
    # Check if all required files are found
    if image_file and bone_file and pred_file:
        print(f"Found all matching files for {image_id}:", flush=True)

        print(f"  Image: {os.path.basename(image_file)}", flush=True)
        print(f"  Lesion: {os.path.basename(lesion_file)}", flush=True)
        print(f"  Bone: {os.path.basename(bone_file)}", flush=True)
        print(f"  Pred: {os.path.basename(pred_file)}", flush=True)

        image_files = {
            'image': image_file,
            'lesion': lesion_file,
            'bone': bone_file,
            'pred': pred_file,
        }
    
    bone_mets_obj = BoneMetastasis(
        image=image_files['image'],
        lesion=image_files['lesion'],
        bone=image_files['bone'],
        pred=image_files['pred']
    )

    bone_mets_obj.lookup_table()
    bone_mets_obj.add_voxel_volume()
    bone_mets_obj.add_priority("ground")
    bone_mets_obj.add_priority("pred")
    bone_mets_obj.add_high_risk()
    bone_mets_obj.merged_table = bone_mets_obj.merge_tables()
    bone_mets_obj.metrics = bone_mets_obj.calculate_metrics()

    return bone_mets_obj