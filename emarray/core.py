import numpy as np


class FieldProperties:
    """
    
    An abstract class which performs physics conversions
    
    """
    mu_0        = 4.0*np.pi*1e-7
    omega_const = 2*np.pi
    def __init__(self, res_convention ='default'):
        """
        
        Parameters
        ==========

        res_convention : str
            the convention for apparent resistivity. May be either
            * Default: rho_ij = |Zij|^2 / (mu omega)
            * Cagniard: rho_ij = 0.2 * T * |Zij|^2
        
        """
        self.res_convention = res_convention
        

    def _get_app_convention(self, freq):
        if self.res_convention =='default':
            const = 1.0 / (self.mu_0*self.omega_const*freq)
        elif self.res_convention =='cagniard':
            const = 0.2 / freq
        return const

    def _get_freq_index(self, freq, index):
        """
        produces an array slice by index, by frequency value, or all frequencies
        
        """
        if freq is None and index is None:
            index = slice(len(self.frequencies))
        elif freq is not None:
            index = np.argwhere(freq==self.frequencies)[0,0]
        return index

    def app_res(self, i, j, freq=None, index=None, log10=False):
        """
        calculates the apparent resistivity

        Parameters
        ==========
        i : int
            Electric field direction. 0 for x, 1 for y

        j : int
            Magnetic field direction. 0 for x, 1 for y

        freq : float or np.ndarray[float] or None
            Frequency at which to query the data from. If a float, TFArray will determine which index corresponds to
            the given frequency. If an np.ndarray, will convert the frequency array to a corresponding index array. If
            None, will either fall back on the provided index keyword argument or return all frequencies

        index : int or np.ndarray[int] or None
            Frequency index at which to query the data from. If an int or np.ndarray of ints, TFArray will 
            slice the frequency array at the given index(es). If None, will either fall back on the provided index
            keyword argument or return all frequencies

        log10 : bool [Optional]
            whether to normalize the output by applying a log10 tranform. Default is False

        Returns
        =======

        float or np.ndarray of floats
            The apparent resistivity
        
        
        """
        index =  self._get_freq_index(freq,index)
        freqs = self.frequencies[index]
        app_res = np.copy(self.impedance[:,index,i,j])
        app_res = app_res * app_res * self._get_app_convention(freqs)[np.newaxis,:,np.newaxis,np.newaxis]
        if log10:
            app_res = np.log10(app_res)
        return np.squeeze(app_res)

    def phase(self, i, j, freq=None, index=None,degrees=True):
        """
        calculates the phase

        Parameters
        ==========
        i : int
            Electric field direction. 0 for x, 1 for y

        j : int
            Magnetic field direction. 0 for x, 1 for y

        freq : float or np.ndarray[float] or None
            Frequency at which to query the data from. If a float, TFArray will determine which index corresponds to
            the given frequency. If an np.ndarray, will convert the frequency array to a corresponding index array. If
            None, will either fall back on the provided index keyword argument or return all frequencies

        index : int or np.ndarray[int] or None
            Frequency index at which to query the data from. If an int or np.ndarray of ints, TFArray will 
            slice the frequency array at the given index(es). If None, will either fall back on the provided index
            keyword argument or return all frequencies

        degrees : bool [Optional]
            whether to normalize the output by converting to degrees. Default is True

        Returns
        =======

        float or np.ndarray of floats
            The phase
        
        
        """
        index =  self._get_freq_index(freq,index)
        phase = np.copy(self.impedance[:,index,i,j])
        angle = np.angle(phase)
        if degrees:
            result = np.rad2deg(angle)
        return np.squeeze(result)

    def invariant(self, freq=None, index=None,type='z1'):
        index =  self._get_freq_index(freq,index)
        z = np.copy(self.impedance)
        if type=='z1':
            inv = (z[:,index, 0, 1] - z[:,index, 1, 0]) / 2.
        elif type=='det':
            inv = np.sqrt(z[:,index, 0, 0]*z[:,index, 1, 1] - z[:,index, 0, 1]*z[:,index, 1, 0])
        elif type=='trace':
            inv =  z[:,index, 0, 0] +  z[:,index, 1, 1]
        elif type=='skew':
            inv = z[:,index,0, 1] - z[:,index, 1, 0]
        elif type=='norm':
            inv = np.sum(np.sum(z[:,index,:,:]*z[:,index,:,:],axis=-1),axis=-1)

        return inv

class EMArray(FieldProperties):
    """
    An object representing one or more MT transfer functions
    
    
    """
    properties = ('tipper','tipper_err','transfer','transfer_err','locations','frequencies')
    def __init__(self, locations, freqs, tipper=True, transfer=True, tipper_err=True, transfer_err=True):
        """
        
        Parameters
        ==========
        locations : list[float] or np.ndarray[float]
            A nx2 or nx3 array of receiver locations where n is the number of receivers. If nx2, the first 
            dimension corresponds to the x coordinate and the second corresponds to the y. If nx3 dimensions,
            the third dimension corresponds to the altitude.

        freqs : list[float] or np.ndarray[float]
            An array of floats corresponding to the frequencies of the transfer function

        tipper : bool [Optional]
            Include a tipper response. Default is True.

        transfer : bool [Optional]
            Include a transfer function. Default is True.

        tipper_err : bool [Optional]
            Include a tipper error array. Default is True.

        transfer_err : bool [Optional]
            Include a transfer error array. Default is True.

        
        """
        super().__init__()
        self.tipper = None
        self.transfer = None
        self.tipper_err = None
        self.transfer_err = None
        self.locations = np.asarray(locations)
        self.frequencies = np.asarray(freqs)
        n_freqs = self.frequencies.shape[0]
        n_rec   = self.locations.shape[0]
        if transfer:
            self.transfer = np.zeros((n_rec,n_freqs,2,2))
        
        if tipper:
            self.tipper   = np.zeros((n_rec,n_freqs,2))

        if transfer_err:
            self.transfer_err = np.zeros((n_rec,n_freqs,2,2))

        if tipper_err:
            self.transfer_err = np.zeros((n_rec,n_freqs,2))

    def add_rx_array(self,*args, tipper=None, transfer=None, tipper_err=None, transfer_err=None,**kwargs):
        """
        add a receiver ensemble to the tfarray

        Parameters:

        tipper : np.ndarray
            a np.ndarray of dimensions n_rec, n_freq, 2. Optional

        transfer : np.ndarray
            a np.ndarray of dimensions n_rec, n_freq, 2, 2. Optional

        tipper_err : np.ndarray
            a np.ndarray of dimensions n_rec, n_freq, 2. Optional

        transfer_err : np.ndarray
            a np.ndarray of dimensions n_rec, n_freq, 2, 2. Optional
        
        
        """
        if tipper is not None:
            self.tipper = tipper
        if transfer is not None:
            self.transfer = transfer
        if tipper_err is not None:
            self.tipper_err = tipper_err
        if transfer_err is not None:
            self.transfer_err = transfer_err
        
    def save_npz(self,file):
        """
        saves tfarray as a .npz file to disk
        
        """
        arr_dict = {}
        for word in self.properties:
            result = getattr(self,word)
            if result is not None:
                arr_dict[word]=result
        np.savez_compressed(file,**arr_dict)

    def load_npz(self,file):
        """
        loads up a tfarray object with data from a .npz file. 
        
        """
        with np.load(file) as data:
            keys = data.keys()
            for key in keys:
                setattr(self,key,data[key])
           

