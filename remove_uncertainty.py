
import numpy as np
import glob 
# load npy file


PATH = f"../../nnUNet_data/nnUNet_processed/Dataset999_AutoPet/nnUNetPlans_3d_fullres/*psma*[0-9].npy"
# load with numpy 
globbed_files = glob.glob(PATH)
print(f"Found {len(globbed_files)} files")
for file in glob.glob(PATH):
    data = np.load(file, allow_pickle=True)
    print(file, data.shape  )
    if data.shape[0] == 3:
        print("Uncertainty map found, removing it")
        ndata = data[0:2,:,:,:]
        print("New shape:", ndata.shape)
        np.save(file, ndata)

