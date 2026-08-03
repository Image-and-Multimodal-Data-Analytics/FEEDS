import numpy as np
import nibabel as nib

from files import find_matching_file, find_matching_files


def load_uncertainty_maps(lesion_id, uncertainty_dir):
    """Load uncertainty maps for a given lesion ID"""
    uncertainty_files = {
        'aleatoric': find_matching_file(lesion_id, uncertainty_dir, "*aleatoric*.nii.gz"),
        'epistemic': find_matching_file(lesion_id, uncertainty_dir, "*epistemic*.nii.gz"),
        'total': find_matching_file(lesion_id, uncertainty_dir, "*total*.nii.gz")
    }
    uncertainty_data = {}
    for unc_type, file_path in uncertainty_files.items():
        if file_path:
            try:
                uncertainty_data[unc_type] = nib.load(file_path).get_fdata()
                print(f"Loaded {unc_type} uncertainty from: {file_path}")
            except Exception as e:
                print(f"Error loading {unc_type} uncertainty: {e}")
                uncertainty_data[unc_type] = None
        else:
            uncertainty_data[unc_type] = None
    
    return uncertainty_data

def align_uncertainty_dimensions(uncertainty_data, reference_shape):
    """Align uncertainty data dimensions with reference shape"""
    aligned_data = {}
    for unc_type, data in uncertainty_data.items():
        if data is not None:
            if np.shape(data) != reference_shape:
                print(f"Reshaping {unc_type} uncertainty from {np.shape(data)} to {reference_shape}")
                try:
                    # Try transposing first
                    data_transposed = np.transpose(data, (2, 1, 0))
                    if np.shape(data_transposed) == reference_shape:
                        aligned_data[unc_type] = data_transposed
                    else:
                        print(f"Warning: {unc_type} uncertainty shape still doesn't match after transpose")
                        aligned_data[unc_type] = None
                except:
                    print(f"Error transposing {unc_type} uncertainty")
                    aligned_data[unc_type] = None
            else:
                aligned_data[unc_type] = data
        else:
            aligned_data[unc_type] = None
    
    # save a .nii file of the aligned uncertainty data
    # if 'total' in aligned_data and aligned_data['total'] is not None:
    #     total_uncertainty_sitk = sitk.GetImageFromArray(aligned_data['total'])
    #     # copy information from the 
    #     total_uncertainty_sitk.CopyInformation(ref_image)
    #     sitk.WriteImage(total_uncertainty_sitk, f"./uncertainty_map/{lesion_id}_total_uncertainty.nii.gz")

    return aligned_data