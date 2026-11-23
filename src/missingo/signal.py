import numpy as np

class Signal:
    def __init__(self, data, fs):

        if not isinstance(data, (list, np.ndarray)):
            raise ValueError("data must be a list or a numpy array")

        arr = np.asarray(data)

        if arr.size == 0:
            raise ValueError("data must not be empty")
        if arr.ndim != 1:
            raise ValueError("data must be a 1D array")
        for value in data:
                    if isinstance(value, (bool, np.bool_)):
                        raise ValueError("data must be numeric (float or int or complex)")
        if np.issubdtype(arr.dtype, np.bool_):
            raise ValueError("data must be numeric (float or int or complex)")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("data must be numeric (float or int or complex)")
        if not np.all(np.isfinite(arr)):
            raise ValueError("data contains non-finite values (NaN or Inf)")

        
        if isinstance(fs, (bool, np.bool_)):
            raise ValueError("fs must be a number")
        if not isinstance(fs, (int, float, np.integer, np.floating)):
            raise ValueError("fs must be a number")
        if not np.isfinite(fs):
            raise ValueError("fs must be finite")  
        if fs<=0:
            raise ValueError("fs must be greater than 0")
        
        self._data = arr.copy()
        self._fs = float(fs) 

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
        return 1.0/self._fs

    @property
    def duration(self):
        return self._data.size/self._fs

    @property
    def time(self):
        time = np.arange(self._data.size)/self._fs
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

    def validate_arithmetic_input(self , other):
        if isinstance(other, ( bool, np.bool_)):
            raise TypeError("only a Signal or a scalar can be used in arithmetic with a Signal object")
        if not isinstance(other,(int, float, np.integer, np.floating, complex, np.complexfloating, Signal)):
            raise TypeError("only a Signal or a scaler can be added to a Signal object")
            
        if isinstance(other, Signal):
            if self.fs != other.fs :
                raise ValueError("both signal objects must have same fs")
            if self.n_samples != other.n_samples:
                raise ValueError("both signal objects must have equal number of data elements")
            
        else :
                
            if not np.isfinite(other):
                raise ValueError("input must be finite")




    def __add__(self,other):
        #validate inputs
        self.validate_arithmetic_input(other)

        fs = self.fs
        #calculating data for resulting signal
        if isinstance(other,Signal):
            x = self.data + other.data 
        else:
            x = self.data + other

        #generating signal based on calculations
        signal = Signal(x,fs)

        return signal

    def __radd__(self, other):
        return self.__add__(other)

    
    def __sub__(self,other):
        #validate inputs
        self.validate_arithmetic_input(other)

        fs = self.fs
        #calculating data for resulting signal
        if isinstance(other,Signal):
            x = self.data - other.data 
        else:
            x = self.data - other

        #generating signal based on calculations
        signal = Signal(x,fs)

        return signal

    def __rsub__(self, other):
        self.validate_arithmetic_input(other)

        x = other - self.data

        signal = Signal(x, self.fs)

        return signal

    
    def __mul__(self,other):
        #validate inputs
        self.validate_arithmetic_input(other)

        fs = self.fs
        #calculating data for resulting signal
        if isinstance(other,Signal):
            x = self.data * other.data 
        else:
            x = self.data * other

        #generating signal based on calculations
        signal = Signal(x,fs)

        return signal

    def __rmul__(self, other):
        return self.__mul__(other)

    
    def __truediv__(self,other):
        #validate inputs
        self.validate_arithmetic_input(other)
        if isinstance(other, Signal):
            if np.any(other.data == 0):
                raise ZeroDivisionError("Signal in denominator must contain non zero samples")
        else:
            if other == 0:
               raise ZeroDivisionError("scalar in denominator can't be zero")
        fs = self.fs
        #calculating data for resulting signal
        if isinstance(other,Signal):
            x = self.data / other.data 
        else:
            x = self.data / other

        #generating signal based on calculations
        signal = Signal(x,fs)

        return signal

    def __rtruediv__(self, other):

        # Validate inputs
        self.validate_arithmetic_input(other)
        fs = self.fs

        # Check denominator for zero
        if np.any(self.data == 0):
            raise ZeroDivisionError("Signal in denominator must contain non-zero samples")

        # Calculate resulting data
        x = other / self.data

        # Generate resulting Signal
        signal = Signal(x, fs)

        return signal

    
