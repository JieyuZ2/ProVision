import os
import json
import numpy as np
from pathlib import Path
import sys
import pickle
import io
import logging
import tarfile
import h5py

def shard_data(list: list, num_shards: int, shard_index: int):
    """ Splits the data into shards """
    assert num_shards > 0, "Number of shards must be greater than 0"
    assert 0 <= shard_index < num_shards, "Shard index must be less than number of shards"
    return list[shard_index::num_shards]

def load_file_by_ext(file: str, ext: str):
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File {file} not found")
    if not file.endswith(ext):
        raise ValueError(f"File {file} must have extension {ext}")
    
    if ext == '.json':
        return load_json(file)
    elif ext == '.jsonl':
        return load_jsonl(file)
    elif ext in ['.pkl', '.pickle']:
        return load_pickle(file)
    else:
        with open(file, 'r') as f:
            return f.read()

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_jsonl(file):
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]

def load_files_from_dir(directory: str, exts: str | list[str] = None) -> list[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory {directory} not found")
    if exts is None:
        return sorted([os.path.join(directory, file) for file in os.listdir(directory)])

    elif isinstance(exts, str):
        exts = [exts]

    return sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(tuple(exts))])

def load_image_files_from_dir(directory: str) -> list[str]:
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    return load_files_from_dir(directory, exts)

''' tar utils'''
def load_all_ids_from_tar(tar_file):
    with tarfile.open(tar_file, 'r') as tar:
        ids = set([member.name for member in tar.getmembers()])
    return ids

def load_content_from_tar(tar_file, file_id):
    with tarfile.open(tar_file, 'r') as tar:
        try:
            member = tar.getmember(file_id)
            f = tar.extractfile(member)
            ext = Path(f).suffix
            content = load_file_by_ext(f, ext)
            return content
        except KeyError:
            print(f"File with id {file_id} not found in the tar archive.")
            return None

def dump_content_to_tar(tar_file, content, filename, extension='json', write_mode='a'):
    # Create an in-memory file object
    file_obj = io.BytesIO()
    
    # Write the content to the in-memory file object
    if extension == 'json':
        file_obj.write(json.dumps(content).encode('utf-8'))
    elif extension == 'jsonl':
        # Handle JSON Lines: each dictionary in content as a separate line
        for item in content:
            line = json.dumps(item) + '\n'  # Convert dict to JSON and add newline
            file_obj.write(line.encode('utf-8'))
    elif extension in ['pickle', 'pkl']:
        pickle.dump(content, file_obj)
    else:
        file_obj.write(content.encode('utf-8'))
    
    # Ensure the file pointer is at the beginning of the file
    file_obj.seek(0)
    
    # Create a TarInfo object with the appropriate name and size
    tar_info = tarfile.TarInfo(name=f"{filename}.{extension}")
    tar_info.size = file_obj.getbuffer().nbytes
    
    # Append the in-memory file object to the tar file
    with tarfile.open(tar_file, write_mode) as tar:
        tar.addfile(tarinfo=tar_info, fileobj=file_obj)

''' hdf5 utils '''
def is_hdf5_file(file):
    return file.endswith('.h5') or file.endswith('.hdf5')

def save_results_to_hdf5(results: list[dict], output_file, 
                         group_key='id', dtype='S', write_mode='a'):
    if dtype == 'dt':
        dtype = h5py.special_dtype(vlen=str)
    with h5py.File(output_file, write_mode) as f:
        for result in results:
            id = result[group_key]
            if id in f: # overwrite existing data
                del f[id]
            grp = f.create_group(id)
            for key, value in result.items():
                grp.create_dataset(key, data=np.array(value, dtype=dtype))

def check_processed_ids_from_hdf5(output_file) -> set[str]:
    if not os.path.exists(output_file):
        logging.info(f"Output file {output_file} does not exist. Returning empty set.")
        return set()

    with h5py.File(output_file, 'r') as f:
        keys_n_vals = [(k, len(f[k])) for k in f.keys()]
        # keep keys with same number of values
        common_keys = [k for k, v in keys_n_vals if v == keys_n_vals[0][1]]
        return set(common_keys)

def load_hdf5_file(hdf5_file: str | h5py.File) -> dict:
    ''' Loads data from h5py file '''
    
    data = {}
    if not isinstance(hdf5_file, h5py.File):
        with h5py.File(hdf5_file, 'r') as f:
            for key, value in f.items():
                data[key] = load_hdf5_group(value)
    else:
        for key, value in hdf5_file.items():
            data[key] = load_hdf5_group(value)
        
    return data

def load_hdf5_group(hdf5_group: h5py.Group) -> dict:
    ''' Loads data from h5py group (not h5py.File or h5py.Dataset) '''
    if type(hdf5_group) is not h5py.Group:
        raise ValueError(f"Input has type {type(hdf5_group)} instead of h5py.Group")
    data = {}
    for key, value in hdf5_group.items():
        if isinstance(value, h5py.Group): # if h5py group, recursively load data
            group_data: dict = load_hdf5_group(value)
            for k, v in group_data.items():
                data[f"{key}/{k}"] = v
        else:
            val = value[()]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            elif isinstance(val, np.ndarray):
                val = val.tolist()
                val = [v.decode('utf-8') if isinstance(v, bytes) else v for v in val]
            data[key] = val
    return data
