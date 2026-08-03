from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    print('I am insdide converter', flush=True)
    print(npz_file)
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    uncer_npy = npz_file[:-4] + "_uncer.npy"
    prob_npy = npz_file[:-4] + "_prob.npy"
    print(data_npy)
    print(seg_npy)
    print(uncer_npy)
    print(prob_npy)
    patient_id = npz_file.split('.')[-2]
    print(patient_id)
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!", flush=True)
                raise e
            #Uncertainty is in the third element, probability is in the fourth element
            #np.save(data_npy, npz_content['data'][:2])
            np.save(data_npy, npz_content['data'])
            #np.save(uncer_npy, npz_content['data'][2])
            #np.save(prob_npy, npz_content['data'][3])


        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!", flush=True)
                raise e
            np.save(npz_file[:-4] + "_seg.npy", npz_content['seg'])

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...", flush=True)
                if fail_ctr < 2:
                    _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1)
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

    except KeyboardInterrupt:
        print('Failed', flush=True)
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes,
                   verify_npy: bool = False):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    npz_files = subfiles(folder, True, None, ".npz", True)
    for npz_file in npz_files:
        _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy)
    
    #print('successfully unpacked', flush= True)

    #with multiprocessing.get_context("spawn").Pool(num_processes) as p:
    #    npz_files = subfiles(folder, True, None, ".npz", True)
    #    print("Number of npz files:", len(npz_files), flush=True)
    #    p.starmap(_convert_to_npy, zip(npz_files,
    #                                   [unpack_segmentation] * len(npz_files),
    #                                   [overwrite_existing] * len(npz_files),
    #                                   [verify_npy] * len(npz_files))
    #              )


def get_case_identifiers(folder: str) -> List[str]:
    """
    finds all npz files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')
