import numpy as np
import metpy.calc as calc
import sympy
from metpy.units import units
import intake
import fsspec
import sys
import xarray as xr
from dask.diagnostics import ProgressBar
import gc
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import matplotlib.pyplot as plt
from iris.coords import DimCoord
from iris.cube import Cube
import iris

L=2264.705*1000 #(j kg-1)
cpdry=1.0035*1000        #(j kg-1 K-1)

'''
Buoyancy calculation based on Williams & Pierrehumbert (2017)
Williams, I. N., and Pierrehumbert, R. T. (2017), Observational evidence against strongly stabilizing tropical cloud feedbacks, Geophys. Res. Lett., 44, 1503– 1510, doi:10.1002/2016GL072202.
'''

def dqdT_calc2(Tdata,pdata):
    T,p = sympy.symbols('T p')
    es=6.112*sympy.exp(17.67*T/(T+243.5))
    #q=621.97*es/(p-es)
    q=0.622*es/(p-es)
    
    dqdT=sympy.diff(q,T)
    dqdtfunc=sympy.lambdify([T,p],dqdT,'numpy')
    dqdTdata=dqdtfunc(Tdata-273.15,pdata/100.)           #convert T to celsius for es equ.
    return(dqdTdata)

def gamma_calc(T,p):
    dqdT=dqdT_calc2(np.array(T),np.array(p))
    gamma=L*dqdT/cpdry
    return(gamma)

def get_zweights(nc):
    if np.shape(nc.plev)[0]!=1:
        pbounds=np.concatenate([[nc.plev.values[0]],0.5*(nc.plev.values[:-1]+nc.plev.values[1:]),[nc.plev.values[-1]]])
        plev_thick=xr.DataArray(np.abs(np.diff(pbounds)),coords={'plev':nc.plev.values})
        plev_thick.name='weights'
        return(plev_thick)
    else:
        plev_thick=xr.DataArray([1.],coords={'plev':nc.plev.values})
        plev_thick.name='weights'
        return(plev_thick)
        
def save_B(var_xr):
    '''
    var_xr
        should be an xarray dataset including:
            'ta'    -   air temperature on pressure levels (K)
            'hus'   -   specific humidity on pressure levels (unitless)
            'zg'    -   geopotential height on pressure levels (m)
            'tas'   -   near surface air temperature (K)
            'ps'    -   surface air pressure (Pa)
    '''
    # split into boundary laye r and free troposphere
    bl_vars=var_xr.metpy.sel(plev=slice(100000.,92500.))
    ft_vars=var_xr.metpy.sel(plev=slice(92500.,30000.))

    bl_parsed=bl_vars.metpy.parse_cf()
    ft_parsed=ft_vars.metpy.parse_cf()

    # calculate h0
    q=calc.mixing_ratio_from_specific_humidity(bl_parsed['hus'])
    z=calc.geopotential_to_height(bl_parsed['zg']*9.8*units('(m**2)*(s**-2)'))

    h=calc.moist_static_energy(z,bl_parsed['ta']*units('K'),q)
    h0_avg=h.weighted(get_zweights(h)).mean('plev')
    
    # calculate h*
    z=calc.geopotential_to_height(ft_parsed['zg']*9.8*units('(m**2)*(s**-2)'))
    p=xr.broadcast(ft_parsed['ta'],ft_parsed['ta'].plev)[1]*units('Pa')
    qstar=calc.saturation_mixing_ratio(p, ft_parsed['ta']*units('K'))
    hstar=calc.moist_static_energy(z,ft_parsed['ta']*units('K'),qstar)
    hstar_avg=hstar.weighted(get_zweights(hstar)).mean('plev')

    ps=var_xr.ps*units('pascal')
    Ts=var_xr.tas*units('kelvin')

    #final buoyancy calculation
    B=(h0_avg-hstar_avg)/((cpdry/1000.)*Ts*(1+gamma_calc(Ts,ps)))
    B=B.drop_vars('metpy_crs').compute()
    
    #save B
    #note some older files may have name='b'
    B.copy(data=np.array(B.data)).to_dataset(name='buoyancy').to_netcdf(path='./output/B.nc')
        

#if __name__ == "__main__":
    #var_xr = ......LOAD
    #save_B(var_xr)