import numpy as np
from scipy.interpolate import griddata

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
            const =  self.mu_0*self.omega_const*freq
        elif self.res_convention =='cagniard':
            const = 5*freq
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
        zij = np.abs(np.copy(self.transfer[:,index,i,j]))
        app_res = zij*zij / self._get_app_convention(freqs)
        if log10:
            app_res = np.log10(app_res)
        return np.squeeze(app_res)

    def phase(self, i, j, freq=None, index=None,deg=True):
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
        phase = np.copy(self.transfer[:,index,i,j])
        angle = np.angle(phase)
        if deg:
            result = np.rad2deg(angle)
        return np.squeeze(result)

    def invariant(self, freq=None, index=None,type='z1'):
        index =  self._get_freq_index(freq,index)
        z = np.copy(self.transfer)
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

class TFArray(FieldProperties):
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
           

class EH1Array:
    """
    an array for storing Eij and Hij values at each receiver position for a single frequency

    """
    def __init__(self,locations,freqs):
        self.locations = np.asarray(locations)
        self.freqs = freqs
        
    def assign_fields(self, E_fields, H_fields):
        """
        Parameters
        ==========

        E_fields: np.ndarray
            a n_rec x n_field_type x n_indicent x n_measured complex array representing the E fields at each receiver location.
            n_field_type: primary (0 index) and secondary (1 index)
            n_incident: x (0 index), y (1 index)
            n_measured: x (0 index), y (1 index)
        H_fields: np.ndarray
            a n_rec x n_field_type x n_indicent x n_measured complex array representing the E fields at each receiver location.
            n_field_type: primary (0 index) and secondary (1 index)
            n_incident: x (0 index), y (1 index)
            n_measured: x (0 index), y (1 index), z (2 index)

        """
        self.E_fields = E_fields
        self.H_fields = H_fields


    def get_field_type(self,scalar_type='abs',**kwargs):
        """
        returns a magnitude measurement of the field

        Parameters
        ==========

        scalar_type : str
            can be absolute value 'abs', real 'real', or imaginary 'imag. 

        field : str
            Either 'e' for Electric or 'h' for Magnetic

        type : str
            Either 'secondary' for secondary or 'primary' for primary

        source : str
            Source direction for fields, either 'x' or 'y'
        
        obs : str
            Observation direction for fields, either 'x' or 'y' for E fields, but can also be
            'z' for magnetic fields

        Returns
        =======
        the absolute value of the fields with the provided parameters

        """
        field = self._field_access(**kwargs)
        if scalar_type=='abs':
            return np.abs(field)
        elif scalar_type=='real':
            return np.real(field)
        elif scalar_type=='imag':
            return np.imag(field)
        else:
            print(f'scalar_type \"{scalar_type}\" is invalid option')

    def get_field_phase(self,degtype='deg',theta_mult=1.0,**kwargs):
        """
        gets the angle of the field strength in the imaginary plane.

        Parameters
        ==========

        field : str
            Either 'e' for Electric or 'h' for Magnetic

        type : str
            Either 'secondary' for secondary or 'primary' for primary

        source : str
            Source direction for fields, either 'x' or 'y'
        
        obs : str
            Observation direction for fields, either 'x' or 'y' for E fields, but can also be
            'z' for magnetic fields

        degtype : str
            whether to return the result in degrees, radians, or sine feature. Can be either 'deg', 'rad', or 'sin'

        theta_mult : float [Optional]
            if degtype='sine', the theta_mult keyword-argument can be used to increase the frequency of 
            the returned sin feature. This is useful when exploring sudden tears or jumps in spatial phase.
        Returns
        =======
        the angle of the fields with the provided parameters

        """
        field = self._field_access(**kwargs)
        if degtype=='deg':
            return np.angle(field,deg=True)
        elif degtype=='rad':
            return np.angle(field)
        elif degtype=='sin':
            return np.sin(np.angle(field)*theta_mult)
        else:
            print(f'degtype \"{degtype}\" is invalid option')

    def _field_access(self,field='e',type='secondary',source='x',obs='x',**kwargs):
        i_type  = int(type!='secondary')
        i_source = int(source!='x')
        if obs=='z':
            i_obs = 2
        else:
            i_obs = int(obs!='x')

        if field=='e':
            return self.E_fields[:,0,i_type,i_source,i_obs]
        else:
            return self.H_fields[:,0,i_type,i_source,i_obs]

    def interp_field(self,xi,yi,scalar_type='abs',**kwargs):
        """
        interpolates the field vaue at xi, yi coordinates

        Parameters
        ==========

        scalar_type : str
            can be any of 'abs', 'real', 'imag', or 'phase'.

        field : str
            Either 'e' for Electric or 'h' for Magnetic

        type : str
            Either 'secondary' for secondary or 'primary' for primary

        source : str
            Source direction for fields, either 'x' or 'y'
        
        obs : str
            Observation direction for fields, either 'x' or 'y' for E fields, but can also be
            'z' for magnetic fields

        degtype : str [Optional]
            whether to return the result in degrees, radians, or sine feature. Can be either 'deg', 'rad', or 'sin'

        theta_mult : float [Optional]
            if degtype='sine', the theta_mult keyword-argument can be used to increase the frequency of 
            the returned sin feature. This is useful when exploring sudden tears or jumps in spatial phase.

        Returns
        =======
            zi : np.ndarray
            a 1-d array of scalar values corresponding to the interpolated field value at that point.

        """
        if scalar_type=='phase':
            field = self.get_field_angle(**kwargs)
        else:
            field = self.get_field_type(scalar_type=scalar_type,**kwargs)

        new_points = np.stack([xi,yi],axis=1)
        interped_field = griddata(self.locations, np.squeeze(field),new_points)
        return interped_field


    def create_tfarray(self):
        tfarray = TFArray(self.locations, self.freqs, tipper=True, transfer=True)

        total_e = self.E_fields[:,:,0,:,:]+self.E_fields[:,:,1,:,:]
        total_h = self.H_fields[:,:,0,:,:]+self.H_fields[:,:,1,:,:]

        target_shape = [total_e.shape[0], total_e.shape[1],total_e.shape[3],total_e.shape[4]]
        total_e = np.reshape(total_e, target_shape)

        h_divisor = total_h[:,:,0,0]*total_h[:,:,1,1] - total_h[:,:,1,0]*total_h[:,:,0,1]

        zxx       = total_e[:,:,0,0]*total_h[:,:,1,1] - total_e[:,:,1,0]*total_h[:,:,0,1]
        zxy       = total_e[:,:,1,0]*total_h[:,:,0,0] - total_e[:,:,0,0]*total_h[:,:,1,0]
        zyx       = total_e[:,:,0,1]*total_h[:,:,1,1] - total_e[:,:,1,1]*total_h[:,:,0,1]
        zyy       = total_e[:,:,1,1]*total_h[:,:,0,0] - total_e[:,:,0,1]*total_h[:,:,1,0]

        kxz       = total_h[:,:,0,2]*total_h[:,:,1,1] - total_h[:,:,1,2]*total_h[:,:,0,1]
        kyz       = total_h[:,:,1,2]*total_h[:,:,0,0] - total_h[:,:,0,2]*total_h[:,:,1,0]

        transfer = np.zeros((target_shape[0],target_shape[1],2,2),dtype=np.complex64)
        tipper   = np.zeros((target_shape[0],target_shape[1],2),dtype=np.complex64)

        transfer[:,:,0,0]=zxx/h_divisor
        transfer[:,:,0,1]=zxy/h_divisor
        transfer[:,:,1,0]=zyx/h_divisor
        transfer[:,:,1,1]=zyy/h_divisor

        tipper[:,:,0]=kxz/h_divisor
        tipper[:,:,1]=kyz/h_divisor

        tfarray.add_rx_array(self,tipper=tipper, transfer=transfer)

        return tfarray

    