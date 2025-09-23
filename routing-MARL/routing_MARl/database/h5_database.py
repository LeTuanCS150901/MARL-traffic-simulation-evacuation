import h5py
import numpy as np
import os
import re
import glob

from logs.logger import logger

class OfflineRLDataset:
    def __init__(self, filename: str, state_dim: int, action_dim: int):
        """
        Create or open an offline RL dataset in HDF5 format.
        If the file exists, create a new one with suffix _1, _2, etc.

        Args:
            filename (str): path to the HDF5 file
            state_dim (int): dimension of state vector
            action_dim (int): dimension of action vector
        """
        self.filename = self._get_unique_filename(filename)
        self.state_dim = state_dim
        self.action_dim = action_dim # Note: action_dim is not used in this version
        self.idx = 0

        # If file doesn't exist, create it with resizable datasets
        if not os.path.exists(self.filename):
            with h5py.File(self.filename, "w") as f:
                maxshape = (None,)
                f.create_dataset("states", (0, state_dim), maxshape=(None, state_dim), dtype="float32")
                # CHANGED: dtype for actions is now 'int64' to match the data being saved.
                f.create_dataset("actions", (0,), maxshape=maxshape, dtype="int64") 
                f.create_dataset("rewards", (0,), maxshape=maxshape, dtype="float32")
                f.create_dataset("next_states", (0, state_dim), maxshape=(None, state_dim), dtype="float32")
                f.create_dataset("group_id", (0,), maxshape=maxshape, dtype=h5py.string_dtype(encoding='utf-8'))
                f.create_dataset("dones", (0,), maxshape=maxshape, dtype="bool")

        logger.success(f"Dataset for offline training created at: {self.filename}")

    def _get_unique_filename(self, filename: str) -> str:
        """
        Return `filename` if it doesn't exist. If it exists, append or increment
        a numeric suffix like _1, _2, ... so the returned path is unique.

        Examples:
        - if "offline_campfire_data_Astar" exists -> "offline_campfire_data_Astar_1"
        - if "offline_campfire_data_Astar_1" exists -> "offline_campfire_data_Astar_2"
        - preserves directory and extension if provided
        """
        dirpath = os.path.dirname(filename)            # may be '' for current directory
        name = os.path.basename(filename)
        base, ext = os.path.splitext(name)             # ext includes the dot, or is ''

        # If base already ends with _<number>, strip that to get the root
        m = re.match(r'^(.*)_(\d+)$', base)
        if m:
            base_root = m.group(1)
        else:
            base_root = base

        # If the requested filename does not exist, return it unchanged
        if not os.path.exists(filename):
            return filename

        # Look for files like base_root_*.ext in the same directory
        pattern = os.path.join(dirpath, f"{base_root}_*{ext}") if dirpath else f"{base_root}_*{ext}"
        candidates = glob.glob(pattern)

        # Extract numeric suffixes from candidate filenames
        indices = []
        regex = re.compile(rf'^{re.escape(base_root)}_(\d+){re.escape(ext)}$')
        for cand in candidates:
            cand_name = os.path.basename(cand)
            mm = regex.match(cand_name)
            if mm:
                indices.append(int(mm.group(1)))

        # If none found, start with 1; otherwise increment the max
        next_idx = max(indices) + 1 if indices else 1

        new_name = f"{base_root}_{next_idx}{ext}"
        return os.path.join(dirpath, new_name) if dirpath else new_name

    def append(self, state, action, reward, next_state, group_id, done):
        """Append one transition to the dataset."""
        with h5py.File(self.filename, "a") as f:
            # The .items() iteration order is not guaranteed in older Python.
            # It's better to access keys directly for clarity.
            
            # Prepare data with correct shapes and types
            state_val = np.asarray(state, dtype="float32").reshape(1, -1)
            # CHANGED: Wrap scalar in a list to make it a 1D array, not 2D
            action_val = np.asarray([action], dtype="int64") 
            reward_val = np.asarray([reward], dtype="float32")
            next_state_val = np.asarray(next_state, dtype="float32").reshape(1, -1)
            # CHANGED: Wrap scalar in a list to make it a 1D array
            group_id_val = np.asarray([group_id], dtype=h5py.string_dtype(encoding='utf-8'))
            done_val = np.asarray([done], dtype="bool")

            data_dict = {
                "states": state_val,
                "actions": action_val,
                "rewards": reward_val,
                "next_states": next_state_val,
                "group_id": group_id_val,
                "dones": done_val,
            }

            for key, value in data_dict.items():
                dset = f[key]
                dset.resize((dset.shape[0] + 1), axis=0)
                dset[-1] = value

            self.idx += 1
        # logger.debug(f"Appended transition {self.idx} to offline training dataset.")

    def __len__(self):
        # It's more efficient to track the length internally than to open the file each time.
        # However, for robustness, opening the file is fine.
        with h5py.File(self.filename, "r") as f:
            return f["states"].shape[0]