import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
import SimpleITK as sitk


import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, List
import os
from time import sleep


class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        print(model_training_output_dir)
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            
            #print(checkpoint['network_weights'].keys())
            #total_params = 0
            #for key in checkpoint['network_weights'].keys():
            #    print(key)
            #    if str(key).split('.')[-1] == 'weight':
            #        print(torch.numel(checkpoint['network_weights'][key]))
            #        total_params += torch.numel(checkpoint['network_weights'][key])
            #print(total_params)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator_with_TTA(data_iterator, save_probabilities, num_processes_segmentation_export)
    
    
    
    
    
    

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)
        # preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        # # hijack batchgenerators, yo
        # # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # # way we don't have to reinvent the wheel here.
        # num_processes = max(1, min(num_processes, len(input_list_of_lists)))
        # ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
        #                         output_filenames_truncated, self.plans_manager, self.dataset_json,
        #                         self.configuration_manager, num_processes)
        # if num_processes == 0:
        #     mta = SingleThreadedAugmenter(ppa, None)
        # else:
        #     mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
        # return mta

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # changing this part, BIRATAL 
                # prediction = self.predict_logits_from_preprocessed_data(data).cpu()
                print('PREDICTING FROM AUGMENTED DATA')
                prediction, uncertainty_map = self.predict_logits_from_preprocessed_data(data)
                prediction = prediction.cpu()
                uncertainty_map = uncertainty_map.cpu()
                # print shape of prediction
                print(f'prediction shape: {prediction.shape}')

                # save the uncertainty map as a .nii file
                print(f'saving uncertainty map for {os.path.basename(ofile) if ofile is not None else "image"}')
                # save as
                # print the average uncertainty value
                print(f'average uncertainty: {uncertainty_map.mean().item()}')
                
                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                    #                               self.dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                            self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                    print('sending off uncertainty map to background worker for export')
                    if save_probabilities:
                        r.append(
                            export_pool.starmap_async(
                                export_prediction_from_logits,
                                ((uncertainty_map, properties, self.configuration_manager, self.plans_manager,
                                self.dataset_json, ofile + '_uncertainty_map.nii.gz', True),)
                            )
                        )
                    
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(
                    #             prediction, self.plans_manager,
                    #              self.configuration_manager, self.label_manager,
                    #              properties,
                    #              save_probabilities)

                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                self.configuration_manager, self.label_manager,
                                properties,
                                save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

            if isinstance(data_iterator, MultiThreadedAugmenter):
                data_iterator._finish()

            # clear lru cache
            compute_gaussian.cache_clear()
            # clear device cache
            empty_cache(self.device)
        return ret
    
    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    # def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
    #     """
    #     IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
    #     TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

    #     RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
    #     SEE convert_predicted_logits_to_segmentation_with_correct_shape
    #     """
    #     n_threads = torch.get_num_threads()
    #     torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
    #     prediction = None

    #     # can I add the augmentations here? 
        

    #     for params in self.list_of_parameters:

    #         # messing with state dict names...
    #         if not isinstance(self.network, OptimizedModule):
    #             self.network.load_state_dict(params)
    #         else:
    #             self.network._orig_mod.load_state_dict(params)

    #         # why not leave prediction on device if perform_everything_on_device? Because this may cause the
    #         # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
    #         # this actually saves computation time
    #         if prediction is None:
    #             prediction = self.predict_sliding_window_return_logits(data).to('cpu')
    #         else:
    #             prediction += self.predict_sliding_window_return_logits(data).to('cpu')

    #     if len(self.list_of_parameters) > 1:
    #         prediction /= len(self.list_of_parameters)

    #     if self.verbose: print('Prediction done')
    #     torch.set_num_threads(n_threads)
    #     return prediction
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> tuple:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)

        aug_fns = [
            (lambda d: d,                        lambda p: p),                    # original
            (lambda d: torch.flip(d, [-1]),      lambda p: torch.flip(p, [-1])), # flip W
            (lambda d: torch.flip(d, [-2]),      lambda p: torch.flip(p, [-2])), # flip H
            (lambda d: torch.flip(d, [-3]),      lambda p: torch.flip(p, [-3])), # flip D
            (lambda d: self._apply_gaussian_blur_volume(d), lambda p: p),        # blur
            (lambda d: d * 1.15,                 lambda p: p),                    # brightness
        ]

        all_probability_maps = []

        for aug_idx, (aug_fn, inv_fn) in enumerate(aug_fns):
            label = 'original' if aug_idx == 0 else f'aug-{aug_idx}'
            if self.verbose:
                print(f'  [{label}] running inference ...')

            augmented_data = aug_fn(data)

            prediction = None
            for params in self.list_of_parameters:
                if not isinstance(self.network, OptimizedModule):
                    self.network.load_state_dict(params)
                else:
                    self.network._orig_mod.load_state_dict(params)

                if prediction is None:
                    prediction = self.predict_sliding_window_return_logits(augmented_data).to('cpu')
                else:
                    prediction += self.predict_sliding_window_return_logits(augmented_data).to('cpu')

            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)

            prediction = inv_fn(prediction)

            # If shape is (1, C, H, W, D) → squeeze batch dim first
            if prediction.dim() == 5:
                prediction = prediction.squeeze(0)  # → (C, H, W, D)
                print(f"Prediction shape after squeeze: {prediction.shape}", flush=True)

            # Now softmax over class dimension (dim=0)
            probs = torch.softmax(prediction, dim=0)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"WARNING: skipping aug {i} - contains nan/inf")
            else: 
                all_probability_maps.append(probs)

        # check if there are nan's in any of the probabiltiy matps
        for i, probs in enumerate(all_probability_maps):
            if torch.isnan(probs).any():
                print(f"NaN found in probability map {i}")
                print(f"This is augmentation {list(aug_fns)[i][0].__name__}")
            if torch.isinf(probs).any():
                print(f"Inf found in probability map {i}")
                print(f"This is augmentation {list(aug_fns)[i][0].__name__}")
            else: 
                print(f"No NaN or Inf found in probability map {i}")
                
        stacked = torch.stack(all_probability_maps, dim=0)
        mean_probs = stacked.mean(dim=0)
        mean_logits = torch.log(mean_probs.clamp(min=1e-8))
        
        # check the dims of mean_probs
        print(f"mean_probs shape: {mean_probs.shape}", flush=True)
        print(f"mean_probs sum over classes: {mean_probs.sum(dim=0).mean()}", flush=True)  # Should be ~1.0
        print(f"mean_probs range: {mean_probs.min()}, {mean_probs.max()}", flush=True)
        
        uncertainty_map = self.calculate_uncertainty(all_probability_maps)

        if self.verbose:
            print('Prediction done')
        torch.set_num_threads(n_threads)

        return mean_logits, uncertainty_map

    def calculate_uncertainty(self, probability_maps: List[torch.Tensor],
                            method: str = 'entropy') -> torch.Tensor:
        """
        Portable uncertainty calculation function that can be swapped out.

        Args:
            probability_maps: List of probability tensors, each of shape (C, *spatial_dims)
                            where C is number of classes. All must be in the same
                            spatial coordinate system.
            method: Uncertainty method to use.
                    Options: 'entropy', 'variance', 'mutual_information'

        Returns:
            uncertainty_map: Tensor of shape (*spatial_dims) containing per-voxel
                            uncertainty values.
        """
        # Stack: (N_aug, C, *spatial_dims)
        stacked = torch.stack(probability_maps, dim=0)
        print(f"stacked has nan: {torch.isnan(stacked).any()}", flush=True)
        print(f"stacked has inf: {torch.isinf(stacked).any()}", flush=True)
        print(f"stacked min/max: {stacked.min()}, {stacked.max()}", flush=True)

        if method == 'entropy':
            mean_probs = stacked.mean(dim=0).float()  # ← force float32
            mean_probs = mean_probs.clamp(min=1e-8, max=1.0 - 1e-8)
            print(f"mean_probs min after clamp: {mean_probs.min()}", flush=True)
            # If mean_probs had NaN, clamp does NOT remove it
            # NaN survives clamp!
            
            uncertainty = -(mean_probs * mean_probs.log()).sum(dim=0)
            # mean_probs = stacked.mean(dim=0)
            
            # # Check how many zeros we have
            # print(f"num exact zeros: {(mean_probs == 0).sum()}", flush=True)
            # print(f"num near zeros (<1e-8): {(mean_probs < 1e-8).sum()}", flush=True)
            
            # mean_probs = mean_probs.clamp(min=1e-8, max=1.0 - 1e-8)
            # mean_probs = mean_probs / mean_probs.sum(dim=0, keepdim=True)
            
            # log_probs = mean_probs.log()
            
            # # Check for any remaining -inf or nan
            # print(f"any inf in log_probs: {torch.isinf(log_probs).any()}", flush=True)
            # print(f"any nan in log_probs: {torch.isnan(log_probs).any()}", flush=True)
            
            # term = mean_probs * log_probs
            
            # # # 0 * -inf = NaN → replace with 0 (mathematically correct: 0*log(0) = 0)
            # # term = torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
            
            # uncertainty = -term.sum(dim=0)
        elif method == 'variance':
            # Mean per-class variance across augmentations
            uncertainty = stacked.var(dim=0).mean(dim=0)

        elif method == 'mutual_information':
            # MI = H[mean] - mean[H]
            mean_probs = stacked.mean(dim=0).clamp(min=1e-10, max=1.0)
            entropy_of_mean = -(mean_probs * mean_probs.log()).sum(dim=0)

            clamped = stacked.clamp(min=1e-10, max=1.0)
            individual_entropies = -(clamped * clamped.log()).sum(dim=1)  # (N_aug, *spatial)
            mean_of_entropies = individual_entropies.mean(dim=0)

            uncertainty = entropy_of_mean - mean_of_entropies

        else:
            raise ValueError(
                f"Unknown uncertainty method: {method}. "
                f"Choose from 'entropy', 'variance', 'mutual_information'"
            )

        return uncertainty


    def apply_tta_augmentation(self, data: torch.Tensor,
                            aug_index: int) -> tuple:
        """
        Apply a specific augmentation to the *full preprocessed volume* before it
        enters the patched/sliding-window prediction pipeline.

        Args:
            data: Full preprocessed tensor of shape (C, *spatial_dims)
            aug_index: 0 = original (identity), 1-5 = augmented versions

        Returns:
            augmented_data: Augmented tensor (same shape)
            inverse_fn: Callable that reverses the *spatial* component of the
                        augmentation on the predicted logits / probabilities
        """

        if aug_index == 0:
            # ---- identity ----
            return data, lambda x: x

        elif aug_index == 1:
            # ---- flip last axis (typically left-right / W) ----
            augmented = torch.flip(data, dims=[-1])
            return augmented, lambda x: torch.flip(x, dims=[-1])

        elif aug_index == 2:
            # ---- flip second-to-last axis (typically anterior-posterior / H) ----
            augmented = torch.flip(data, dims=[-2])
            return augmented, lambda x: torch.flip(x, dims=[-2])

        elif aug_index == 3:
            # ---- flip depth axis for 3-D; combined H+W flip for 2-D ----
            if data.dim() >= 4:                         # (C, D, H, W)
                augmented = torch.flip(data, dims=[-3])
                return augmented, lambda x: torch.flip(x, dims=[-3])
            else:                                        # (C, H, W)
                augmented = torch.flip(data, dims=[-1, -2])
                return augmented, lambda x: torch.flip(x, dims=[-1, -2])

        elif aug_index == 4:
            # ---- Gaussian blur ----
            kernel_size = 3
            sigma = 0.8
            coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
            g1d = g1d / g1d.sum()

            num_channels = data.shape[0]

            if data.dim() == 4:                          # 3-D volume
                kernel = (g1d[:, None, None]
                        * g1d[None, :, None]
                        * g1d[None, None, :])
                kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1, 1)
                augmented = F.conv3d(
                    data.unsqueeze(0).float(), kernel,
                    padding=kernel_size // 2, groups=num_channels
                ).squeeze(0).to(data.dtype)
            else:                                        # 2-D image
                kernel = g1d[:, None] * g1d[None, :]
                kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)
                augmented = F.conv2d(
                    data.unsqueeze(0).float(), kernel,
                    padding=kernel_size // 2, groups=num_channels
                ).squeeze(0).to(data.dtype)

            # Blur has no spatial inverse on prediction side
            return augmented, lambda x: x

        elif aug_index == 5:
            # ---- Intensity / brightness scaling ----
            factor = 1.15
            augmented = data * factor
            return augmented, lambda x: x

        else:
            raise ValueError(f"Unknown augmentation index: {aug_index}")


    def predict_from_data_iterator_with_TTA(
            self,
            data_iterator,
            save_probabilities: bool = False,
            num_processes_segmentation_export: int = default_num_processes,
            uncertainty_method: str = 'entropy',
            save_uncertainty: bool = True,
            uncertainty_output_dir: Optional[str] = None):

        uncertainty_maps = {}

        with multiprocessing.get_context("spawn").Pool(
                num_processes_segmentation_export) as export_pool:

            worker_list = [i for i in export_pool._pool]
            r = []

            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)} with TTA:')
                    print(f'Size of image to predict: {data.shape}')
                else:
                    print(f'\nPredicting image of shape {data.shape} with TTA:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')
  
                properties = preprocessed['data_properties']

                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)



                prediction, uncertainty_map = self.predict_logits_from_preprocessed_data(data)
                prediction = prediction.cpu()
                uncertainty_map = uncertainty_map.cpu()
                print(f"uncertainty min: {uncertainty_map.min()}", flush=True)
                print(f"uncertainty max: {uncertainty_map.max()}", flush=True)
                print(f"uncertainty mean: {uncertainty_map.mean()}", flush=True)
                print(f"len of uncertainty unique vals: {len(uncertainty_map.unique())}", flush=True)
                
                
                print(f"probability map min: {prediction.min()}", flush=True)
                print(f"probability map max: {prediction.max()}", flush=True)
                print(f"probability map mean: {prediction.mean()}", flush=True)
                print(f"len of probability unique vals: {len(prediction.unique())}", flush=True)
                # Save uncertainty as .nii.gz in original image space
                if save_uncertainty and ofile is not None:
                    unc_dir = uncertainty_output_dir or os.path.dirname(ofile)
                    os.makedirs(unc_dir, exist_ok=True)
                    basename = os.path.basename(ofile)
                    if basename.endswith('.nii.gz'):
                        unc_basename = basename.replace('.nii.gz', '_uncertainty.nii.gz')
                    elif basename.endswith('.nii'):
                        unc_basename = basename.replace('.nii', '_uncertainty.nii.gz')
                    elif basename.endswith('.npz'):
                        unc_basename = basename.replace('.npz', '_uncertainty.nii.gz')
                    else:
                        # ofile has no recognized extension, just append
                        unc_basename = basename + '_uncertainty.nii.gz'

                    unc_file = os.path.join(unc_dir, unc_basename)
                    
                    self.save_uncertainty_map_as_nifti(
                        uncertainty_map,
                        properties,
                        self.plans_manager,
                        self.configuration_manager,
                        unc_file
                    )

                map_key = ofile if ofile is not None else f'image_{id(data)}'
                uncertainty_maps[map_key] = uncertainty_map

                if ofile is not None:
                    print(f"Mean uncertainty: {uncertainty_map.mean().item()}")
                    print('Sending TTA prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                            self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('Sending TTA prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape,
                            ((prediction, self.plans_manager, self.configuration_manager,
                            self.label_manager, properties, save_probabilities),)
                        )
                    )

                if ofile is not None:
                    print(f'Done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}')

            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        compute_gaussian.cache_clear()
        empty_cache(self.device)

        return ret, uncertainty_maps
        
    def save_uncertainty_map_as_nifti(self, uncertainty_map: torch.Tensor,
                                    properties: dict,
                                    plans_manager,
                                    configuration_manager,
                                    output_path: str):
        """
        Resample uncertainty map back to original image space and save as .nii.gz

        Args:
            uncertainty_map: tensor of shape (*spatial_dims) in preprocessed space
            properties: data_properties dict from preprocessing (contains spacing, shape, etc.)
            plans_manager: nnUNet plans manager
            configuration_manager: nnUNet configuration manager
            output_path: path to save the .nii.gz file
        """
        # uncertainty_map is (*spatial_dims), we need to add a channel dim for resampling
        # Shape: (1, *spatial_dims)
        uncertainty_np = uncertainty_map.numpy().astype(np.float32)
        uncertainty_np = uncertainty_np[np.newaxis]

        # Get original spacing and new spacing
        original_spacing = properties['spacing']
        shape_original_before_cropping = properties['shape_before_cropping']

        # Resample back to original resolution
        # Use the same resampling the segmentation uses, but with linear interpolation
        # since uncertainty is continuous
        current_spacing = configuration_manager.spacing

        # Compute the shape after cropping but before resampling
        shape_after_cropping = properties.get('shape_after_cropping_and_before_resampling',
                                            shape_original_before_cropping)

        # Resample uncertainty to original spacing
        from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
        uncertainty_resampled = resample_data_or_seg_to_shape(
            uncertainty_np,
            shape_after_cropping,
            current_spacing,
            original_spacing,
            is_seg=False
        )

        # Remove channel dim
        uncertainty_resampled = uncertainty_resampled[0]

        # If the image was cropped during preprocessing, we need to pad back
        # to the original shape
        if 'bbox_used_for_cropping' in properties:
            bbox = properties['bbox_used_for_cropping']
            full_uncertainty = np.zeros(shape_original_before_cropping, dtype=np.float32)
            slicer = tuple(slice(b[0], b[1]) for b in bbox)
            # Handle potential shape mismatches from rounding during resampling
            actual_shape = tuple(min(uncertainty_resampled.shape[i], b[1] - b[0])
                                for i, b in enumerate(bbox))
            crop_slicer = tuple(slice(0, s) for s in actual_shape)
            paste_slicer = tuple(slice(b[0], b[0] + s) for b, s in zip(bbox, actual_shape))
            full_uncertainty[paste_slicer] = uncertainty_resampled[crop_slicer]
            uncertainty_resampled = full_uncertainty

        # Create SimpleITK image
        # Note: SimpleITK uses (x,y,z) ordering, numpy uses (z,y,x)
        sitk_image = sitk.GetImageFromArray(uncertainty_resampled)

        # Set spacing, direction, origin from original image
        sitk_image.SetSpacing([float(s) for s in original_spacing[::-1]])

        if 'sitk_stuff' in properties:
            sitk_stuff = properties['sitk_stuff']
            sitk_image.SetOrigin(sitk_stuff['origin'])
            sitk_image.SetDirection(sitk_stuff['direction'])

        # Save
        sitk.WriteImage(sitk_image, output_path)
        print(f'  Saved uncertainty .nii.gz → {output_path}')
    
    def _apply_gaussian_blur_volume(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gaussian blur on full volume. x shape: (C, D, H, W)
        """
        kernel_size = 3
        sigma = 0.8
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
        g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()

        C = x.shape[0]
        kernel = (g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :])
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1, 1)

        blurred = F.conv3d(
            x.unsqueeze(0).float(), kernel,
            padding=kernel_size // 2, groups=C
        ).squeeze(0).to(x.dtype)

        return blurred

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
                
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                
                
                
                
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
            # and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
            # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
                
                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                               self.perform_everything_on_device)
                    except RuntimeError:
                        print(
                            'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits
    
    def predict_augmented_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose:
                    print(f'Input shape: {input_image.shape}')
                    print("step_size:", self.tile_step_size)
                    print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # Define augmentations: (augment_fn, inverse_fn)
                aug_fns = [
                    (lambda d: d,                        lambda p: p),                         # original
                    (lambda d: torch.flip(d, [-1]),      lambda p: torch.flip(p, [-1])),       # flip W
                    (lambda d: torch.flip(d, [-2]),      lambda p: torch.flip(p, [-2])),       # flip H
                    (lambda d: torch.flip(d, [-3]),      lambda p: torch.flip(p, [-3])),       # flip D
                    (lambda d: self._apply_gaussian_blur_volume(d), lambda p: p),              # blur
                    (lambda d: d * 1.15,                 lambda p: p),                         # brightness
                ]

                all_logits = None

                for i, (aug_fn, inv_fn) in enumerate(aug_fns):
                    if self.verbose:
                        label = 'original' if i == 0 else f'aug-{i}'
                        print(f'  [{label}] running sliding window inference ...')

                    augmented = aug_fn(input_image)

                    data, slicer_revert_padding = pad_nd_image(
                        augmented, self.configuration_manager.patch_size,
                        'constant', {'value': 0}, True, None
                    )

                    slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                    if self.perform_everything_on_device and self.device != 'cpu':
                        try:
                            logits = self._internal_predict_sliding_window_return_logits(
                                data, slicers, self.perform_everything_on_device)
                        except RuntimeError:
                            print('Prediction on device was unsuccessful, falling back to CPU')

                        empty_cache(self.device)
                        logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                    else:
                        logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)

                    # Revert padding
                    logits = logits[(slice(None), *slicer_revert_padding[1:])]

                    # Inverse spatial transform
                    logits = inv_fn(logits)

                    # Accumulate
                    if all_logits is None:
                        all_logits = logits
                    else:
                        all_logits += logits

                # Average across augmentations

                    all_logits /= len(aug_fns)
                    empty_cache(self.device)

        return all_logits

def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', '--c', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                verbose_preprocessing=args.verbose)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)
    print("continue_prediction set to:", args.continue_prediction)
    
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=1, part_id=0)


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the nnUNet_results folder.')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=args.num_parts,
                                 part_id=args.part_id)
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)


if __name__ == '__main__':
    # predict a bunch of files
    predict_entry_point()



    def _apply_gaussian_blur_volume(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gaussian blur on full volume. x shape: (C, D, H, W)
        """
        kernel_size = 3
        sigma = 0.8
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
        g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()

        C = x.shape[0]
        kernel = (g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :])
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1, 1)

        blurred = F.conv3d(
            x.unsqueeze(0).float(), kernel,
            padding=kernel_size // 2, groups=C
        ).squeeze(0).to(x.dtype)

        return blurred