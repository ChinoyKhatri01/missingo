import numpy as np 

class Signal:
    def __init__(self, data, fs):

        arr = np.array(data)

        if arr.size == 0 :
            raise ValueError("data must not be empty")
        if arr.ndim != 1 :
            raise ValueError("data must be a 1D array")
        for value in data:
                    if isinstance(value, (bool, np.bool_)):
                        raise ValueError("data must be numeric (float or int or complex)")
        if np.issubdtype(arr.dtype, np.bool_) :
            raise ValueError("data must be numeric (float or int or complex)")
        if not np.issubdtype(arr.dtype, np.number) :
            raise ValueError("data must be numeric (float or int or complex)")
        if not np.all(np.isfinite(arr)) :
            raise ValueError("data contains non-finite values (NaN or Inf)")

        if isinstance(fs, (bool, np.bool_)):
            raise ValueError("fs must be a number")
        if not isinstance(fs, (int, float, np.integer, np.floating)) :
            raise ValueError("fs must be a number")
        if fs <= 0 :
            raise ValueError("fs must be greater than 0")
        if not np.isfinite(fs) :
            raise ValueError("fs must be finite")
        
        self._data = arr.copy()
        self._fs = fs 

        self._data.flags.writeable = False

    @property
    def data(self):
        return self._data.copy()

    @property
    def fs(self):
        return self._fs

    @property
    def n_samples(self):
        return self._data.size

    @property
    def sampling_interval(self):
        return 1.0 / self._fs

    @property
    def duration(self):
        return self._data.size /self._fs

    @property
    def time(self):
        time = np.arange(self._data.size) / self._fs
        time.flags.writeable = False
        return time

    def __len__(self):
        return self._data.size

    def __repr__(self):
        sig = f"""Signal(n_samples = {self._data.size},
          fs = {self._fs},
          duration = {self.duration},
          dtype = {self._data.dtype},
          ndim = {self._data.ndim})"""
        return sig



