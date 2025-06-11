import numpy as np
import os
from pixell import enmap
import re

def tiny_imagenet_file_handler(source_path, return_path = True):
    train_path = os.path.join(source_path, 'tiny-imagenet-200/train')
    test_path = os.path.join(source_path, 'tiny-imagenet-200/test/images')

    train_class_path = [os.path.join(train_path, class_id + '/images') for class_id in os.listdir(train_path)]
    
    train_image_path = []
    for tp in train_class_path:
        for image_id in os.listdir(tp):
            train_image_path.append(os.path.join(tp, image_id))

    test_image_path = []
    for image_id in os.listdir(test_path):
        test_image_path.append(os.path.join(test_path, image_id))

    print('#train_images: ', len(train_image_path))
    print('#test_images: ', len(test_image_path))

    if return_path:
        return train_image_path, test_image_path
    

def get_cosmo_data(source_path='/home/am3353/Gibbs-Diff/data/cosmo/created_data', n_samples=1000):
    """
    Loads cosmological map data and their corresponding parameters from a directory structure.

    Args:
        source_path (str): The root directory containing 'cmb_maps', 'dust_maps', etc.
        n_samples (int): The number of samples to load.

    Returns:
        tuple: A tuple containing cmb, dust, mixed maps, and parameters, each as a NumPy array.
    """
    
    # --- 1. Find and sort all relevant subdirectories ---
    all_dirs = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    print(f"Found directories: {all_dirs}")
    
    # Separate map directories from parameter directories
    map_dirs = sorted([d for d in all_dirs if 'maps' in d])
    params_dir = os.path.join(source_path, 'params')

    if not map_dirs:
        raise FileNotFoundError(f"No 'maps' directories found in {source_path}")
    if not os.path.exists(params_dir):
        raise FileNotFoundError(f"The 'params' directory was not found in {source_path}")

    maps_dict = {type_dir: [] for type_dir in map_dirs}
    
    # --- 2. Load the map files ---
    for type_dir in map_dirs:
        type_path = os.path.join(source_path, type_dir)
        
        # Get a numerically sorted list of files to ensure order
        files_to_load = sorted(os.listdir(type_path), key=lambda x: int(re.search(r'_(\d+)\.fits$', x).group(1)))
        
        # Limit to n_samples if necessary
        files_to_load = files_to_load[:n_samples]

        for map_file in files_to_load:
            # Use the FULL path to the file for loading
            map_path = os.path.join(type_path, map_file)
            try:
                # enmap.read_map loads the FITS file into an ndmap (NumPy-like array)
                loaded_map = enmap.read_map(map_path)
                maps_dict[type_dir].append(loaded_map)
            except Exception as e:
                print(f"Warning: Could not load or process file {map_path}. Error: {e}")

    # --- 3. Load the parameter file ---
    # Assuming the params file is a single text file (e.g., params.txt)
    # This part may need adjustment based on the actual file format (e.g., .csv, .npy)
    try:
        # Find the first file in the params directory
        params_file_name = os.listdir(params_dir)[0] 
        params_file_path = os.path.join(params_dir, params_file_name)
        # Load as a text file, assuming space-separated values
        params_arr = np.loadtxt(params_file_path)[:n_samples]
    except (FileNotFoundError, IndexError):
        print("Warning: No parameter file found. Returning empty array for params.")
        params_arr = np.array([])
    except Exception as e:
        print(f"Warning: Could not load or process parameter file. Error: {e}")
        params_arr = np.array([])

    # --- 4. Convert lists to NumPy arrays ---
    # The dictionary keys are now sorted, so we can rely on their order
    cmb_arr = np.array(maps_dict['cmb_maps'])
    dust_arr = np.array(maps_dict['dust_maps'])
    mixed_arr = np.array(maps_dict['mixed_maps'])
    # params_arr = np.array(maps_dict['params'])

    print("\n--- Final Shapes ---")
    print(f"CMB shape: {cmb_arr.shape}")
    print(f"Dust shape: {dust_arr.shape}")
    print(f"Mixed shape: {mixed_arr.shape}")

    # --- 5. Return the data with an expanded channel dimension ---
    # Assuming you want to add a "channel" dimension for downstream models (e.g., PyTorch)
    return np.expand_dims(cmb_arr, axis=1), np.expand_dims(dust_arr, axis=1), np.expand_dims(mixed_arr, axis=1), params_arr

if __name__ == '__main__':
    print('running __data_file_handler.py__')

    obs, signal, noise, params = get_cosmo_data()
    print(obs.shape, signal.shape, noise.shape, params.shape)