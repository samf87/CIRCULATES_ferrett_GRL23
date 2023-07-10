import iris,glob
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt
try:
    from iris.util import equalise_attributes
except:pass
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import interp1d
import iris.coord_categorisation as icc
import sys
import glob
import xarray as xr
import intake
import fsspec
import seaborn as sns
import pandas as pd
import datetime
from itertools import product

def regrid(nc,lat=30,res=2):

    latitude = iris.coords.DimCoord(np.arange(-1*lat, lat+1, res),
                     standard_name='latitude',
                     units='degrees')
    latitude.guess_bounds()
    longitude = iris.coords.DimCoord(np.arange(0, 360, res),
                         standard_name='longitude',
                         units='degrees')
    longitude.guess_bounds()

    newgrid = iris.cube.Cube(np.zeros((len(np.arange(-1*lat, lat+1, res)), len(np.arange(0,360,res))), np.float32),
        dim_coords_and_dims=[(latitude, 0),(longitude, 1)])
    
    return(nc.regrid(newgrid,iris.analysis.AreaWeighted(mdtol=0.5)))

def sub_nc(nc,Bbool):
    wgts = iris.analysis.cartography.area_weights(nc)
    nc=nc.copy(np.ma.masked_where(~Bbool,nc.data))
    return(nc.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=wgts).data)

def lin_predict(y,x,x0):
    
    x_const=sm.add_constant(x)
    
    fit=sm.OLS(y,x_const)
    res = fit.fit()
    x0_const = sm.add_constant(np.array([x0]))
    y0 = res.predict([1,x0])

    return y0[0]

def lin_reg(y,x):
    x = np.ravel(x)
    y = np.ravel(y)
    
    if len(x) == 0:
        return([np.nan])

    x = sm.add_constant(x)
    fit = sm.OLS(y,x)
    res = fit.fit()

    cint = res.conf_int(alpha=0.05)

    if len(res.params) == 2:
        intercept = res.params[0]
        slope = res.params[1]
        return {'vals':(slope,intercept),'cint':(slope-cint[1][0],intercept-cint[0][0]),'se':[res.bse[1]]}
    else:
        return {'vals':res.params,'cint': [res.params[0]-cint[0][0]],'se':res.bse}

def bin_by_B_regress2(feed,B,T,bins,lat=30,Tfull=False,err_type='cint'):

    B=B.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))
    feed=feed.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))
    if not Tfull:
        T=T.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))

    wgts = iris.analysis.cartography.area_weights(T)
    Tavg=T.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=wgts).data
    lr = lin_reg(sub_nc(feed,np.ma.filled(np.ma.less(B.data,bins[0]),False)),Tavg)
    feed_avg=[lr['vals'][0]]
    feed_cint=[lr[err_type][0]]
    
    limcheck=lambda a,b: np.ma.filled(np.ma.logical_and(np.ma.greater_equal(B.data,a),np.ma.less(B.data,b)),False)
    lrs = list(map(lambda lim1,lim2: lin_reg(sub_nc(feed,limcheck(lim1,lim2)),Tavg),bins[:-1],bins[1:]))
    feed_avg.extend(map(lambda x: x['vals'][0],lrs))
    feed_cint.extend(map(lambda x: x[err_type][0],lrs))
    
    lr = lin_reg(sub_nc(feed,np.ma.filled(np.ma.greater_equal(B.data,bins[-1]),False)),Tavg)
    feed_avg.extend([lr['vals'][0]])
    feed_cint.extend([lr[err_type][0]])

    return(feed_avg,feed_cint)

def bin_by_B_regress_zero(feed,B,T,lat=30,Tfull=False):

    B=B.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))
    feed=feed.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))

    if not Tfull:
        T=T.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))

    wgts = iris.analysis.cartography.area_weights(T)
    Tavg=T.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=wgts).data
    feed_avg=[lin_reg(sub_nc(feed,np.ma.filled(np.ma.less(B.data,0),False)),Tavg)['vals'][0]]
    feed_cint=[lin_reg(sub_nc(feed,np.ma.filled(np.ma.less(B.data,0),False)),Tavg)['cint'][0]]
    
    feed_avg.extend([lin_reg(sub_nc(feed,np.ma.filled(np.ma.greater_equal(B.data,0),False)),Tavg)['vals'][0]])
    feed_cint.extend([lin_reg(sub_nc(feed,np.ma.filled(np.ma.greater_equal(B.data,0),False)),Tavg)['cint'][0]])

    return(feed_avg,feed_cint)

def longterm_feedback_calc_B(Bs, SWCREs, LWCREs, tsanos, binpc=[55,85],full=False,Tfull=False):
    '''
    Calculate and save abrupt-4xCO2 feedbacks in B bins. Ensure all inputs are on same grid
    or add in regrid function and in iris format

    Input
    ---
    Bs        List of CMIP6 buoyancys
    SWCREs    List of CMIP6 adjusted SWCRE anoms
    LWCREs    List of CMIP6 adjusted LWCRE anoms
    tsanos    List of CMIP6 surface temps anoms
    binpc     List of bins to split regimes
    full      Bool; True if want to calculate full tropics, no regime
    Tfull     Bool; True to use global SSTA in regression, False to use tropical
    '''
    
    ys_4x=[];ys2_4x=[];ys3_4x=[]
    ysz_4x=[];ysz2_4x=[];ysz3_4x=[]
    
    landsea_mask=iris.load_cube('./input/land_sea_mask.nc')
    full_mask = regrid(landsea_mask,lat=90)
    landsea_mask = regrid(landsea_mask)
    landmask=np.where(landsea_mask.data<50,True,False)
    full_landmask=np.where(full_mask.data<50,True,False)

    for B, SWCRE, LWCRE, ts_ano in zip(Bs, SWCREs, LWCREs, tsanos):

        #if inputs not on same grid
        #B,SWCRE,LWCRE,ts_ano = list(map(regrid,[B, SWCRE, LWCRE, ts_ano]))
        
        try:
            icc.add_year(SWCRE,'time')
            icc.add_year(LWCRE,'time')
            icc.add_year(ts_ano_r,'time')
        except:pass
        try:
            icc.add_year(B,'time')
        except:pass

        #mask land
        nmonths = np.shape(SWCRE)[0]
        nyrs = int(nmonths/12)
        SWCRE=SWCRE.copy(np.ma.masked_where([landmask]*nmonths,SWCRE.data))
        LWCRE=LWCRE.copy(np.ma.masked_where([landmask]*nmonths,LWCRE.data))
        B=B.copy(np.ma.masked_where([landmask]*nmonths,B.data))
        if Tfull:
            ts_ano = ts_ano.copy(np.ma.masked_where([full_landmask]*nmonths,ts_ano.data))
        else:
            ts_ano = ts_ano.copy(np.ma.masked_where([landmask]*nmonths,ts_ano.data))

        #yearly average
        SWCRE_yr=SWCRE.aggregated_by('year',iris.analysis.MEAN)
        LWCRE_yr=LWCRE.aggregated_by('year',iris.analysis.MEAN)
        ts_ano_yr=ts_ano.aggregated_by('year',iris.analysis.MEAN)
        B_yr=B.aggregated_by('year',iris.analysis.MEAN)
        B_yr=B_yr.copy(np.ma.masked_where([landmask]*nyrs,B_yr.data))

        #calculate B percentile bins
        fulldata=B_yr.data.data[~B_yr.data.mask]
        Bbins=np.percentile(fulldata,binpc)

        #perform regression to obtain feedbacks
        if full:
            wgts = iris.analysis.cartography.area_weights(ts_ano_yr)
            Tavg=ts_ano_yr.collapsed(['latitude', 'longitude'],iris.analysis.MEAN, weights=wgts)

            wgts = iris.analysis.cartography.area_weights(SWCRE_yr)
            feedavgs = [nc.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=wgts).data 
                for nc in [SWCRE_yr,LWCRE_yr,SWCRE_yr+LWCRE_yr]]

            ys_4x.append([lin_reg(y,Tavg.data)['vals'][0] for y in feedavgs])

        else:
            ys_4x.append(bin_by_B_regress2(SWCRE_yr,B_yr,ts_ano_yr,bins=Bbins,lat=30,Tfull=Tfull))
            ys2_4x.append(bin_by_B_regress2(LWCRE_yr,B_yr,ts_ano_yr,bins=Bbins,lat=30,Tfull=Tfull))
            ys3_4x.append(bin_by_B_regress2(SWCRE_yr+LWCRE_yr,B_yr,ts_ano_yr,bins=Bbins,lat=30,Tfull=Tfull))
            
            ysz_4x.append(bin_by_B_regress_zero(SWCRE_yr,B_yr,ts_ano_yr,lat=30,Tfull=Tfull))
            ysz2_4x.append(bin_by_B_regress_zero(LWCRE_yr,B_yr,ts_ano_yr,lat=30,Tfull=Tfull))
            ysz3_4x.append(bin_by_B_regress_zero(SWCRE_yr+LWCRE_yr,B_yr,ts_ano_yr,lat=30,Tfull=Tfull))
    
    #save feedbacks
    Tadd = '_Tfull' if Tfull else ''
    if full:
        np.save(f'./output/cmip_4xco2_feeds_binnedpic_full_mask_compare{Tadd}.npy',np.transpose(ys_4x))
    else:
        np.save(f'./output/cmip_4xco2_feeds_binnedpic_2lims_{binpc[0]}{binpc[1]}_mask_compare{Tadd}.npy',[ys_4x,ys2_4x,ys3_4x])
        np.save(f'./output/cmip_4xco2_feeds_binnedpic_zero_mask_compare{Tadd}.npy',[ysz_4x,ysz2_4x,ysz3_4x])


def innann_feedback_calc_B(Bs, SWCREs, LWCREs, tss, binpc=[55,85],full=False,Tfull=False):
    '''
    Calculate and save abrupt-4xCO2 feedbacks in B bins.
    Inputs don't need to be anomalies, these are calculated.
    Ensure all inputs are on same grid
    or add in regrid function and in iris format.

    Input
    ---
    Bs        List of CMIP6 buoyancys
    SWCREs    List of CMIP6 adjusted SWCRE
    LWCREs    List of CMIP6 adjusted LWCRE
    tss       List of CMIP6 surface temps
    binpc     List of bins to split regimes
    full      Bool; True if want to calculate full tropics, no regime
    Tfull     Bool; True to use global SSTA in regression, False to use tropical
    '''
    
    ys_4x=[];ys2_4x=[];ys3_4x=[]
    ysz_4x=[];ysz2_4x=[];ysz3_4x=[]
    
    landsea_mask=iris.load_cube('./input/land_sea_mask.nc')
    full_mask = regrid(landsea_mask,lat=90)
    landsea_mask = regrid(landsea_mask)
    landmask=np.where(landsea_mask.data<50,True,False)
    full_landmask=np.where(full_mask.data<50,True,False)

    for B, SWCRE, LWCRE, ts in zip(Bs, SWCREs, LWCREs, tss):

        #if inputs not on same grid
        #B,SWCRE,LWCRE,ts = list(map(regrid,[B, SWCRE, LWCRE, ts]))

        try:
            icc.add_month(ts,'time')
            icc.add_month(SWCRE,'time')
            icc.add_month(LWCRE,'time')
        except:pass

        #create anomalies
        nmonths = np.shape(SWCRE)[0]
        nyrs = int(nmonths/12)

        ts_anom=ts-np.concatenate([ts.aggregated_by('month',iris.analysis.MEAN).data]*nyrs,axis=0)
        SWCRE_anom=SWCRE-np.concatenate([SWCRE.aggregated_by('month',iris.analysis.MEAN).data]*nyrs,axis=0)
        LWCRE_anom=LWCRE-np.concatenate([LWCRE.aggregated_by('month',iris.analysis.MEAN).data]*nyrs,axis=0)

        #land mask
        SWCRE_anom=SWCRE_anom.copy(np.ma.masked_where([landmask]*nmonths,SWCRE_anom.data))
        LWCRE_anom=LWCRE_anom.copy(np.ma.masked_where([landmask]*nmonths,LWCRE_anom.data))
        if not Tfull:
            ts_pi_anom=ts_pi_anom.copy(np.ma.masked_where([landmask]*nmonths,ts_pi_anom.data))
        else:
            ts_pi_anom=ts_pi_anom.copy(np.ma.masked_where([landmask_full]*nmonths,ts_pi_anom.data))

        Bpi=Bpi.copy(np.ma.masked_where([landmask]*nmonths,Bpi.data))

        #calculate percentiles
        fulldata=Bpi.data.data[~Bpi.data.mask]
        Bbins=np.percentile(fulldata,binpc)
                
        wgts = iris.analysis.cartography.area_weights(ts_pi_anom)
        Tavg=ts_pi_anom.collapsed(['latitude', 'longitude'],iris.analysis.MEAN,
            weights=wgts)
        
        #calculate feedbacks
        if full:
            wgts = iris.analysis.cartography.area_weights(SWCRE_anom)
            feedavgs = [nc.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=wgts).data 
                for nc in [SWCRE_anom,LWCRE_anom,SWCRE_anom+LWCRE_anom]]

            ys.append([lin_reg(y,Tavg.data)['vals'][0] for y in feedavgs])
        else:
            ys.append(bin_by_B_regress2(SWCRE_anom,Bpi,ts_pi_anom,bins=Bbins,lat=30,Tfull=Tfull))
            ys2.append(bin_by_B_regress2(LWCRE_anom,Bpi,ts_pi_anom,bins=Bbins,lat=30,Tfull=Tfull))
            ys3.append(bin_by_B_regress2(SWCRE_anom+LWCRE_anom,Bpi,ts_pi_anom,bins=Bbins,lat=30,Tfull=Tfull))
            
            ysz.append(bin_by_B_regress_zero(SWCRE_anom,Bpi,ts_pi_anom,lat=30,Tfull=Tfull))
            ysz2.append(bin_by_B_regress_zero(LWCRE_anom,Bpi,ts_pi_anom,lat=30,Tfull=Tfull))
            ysz3.append(bin_by_B_regress_zero(SWCRE_anom+LWCRE_anom,Bpi,ts_pi_anom,lat=30,Tfull=Tfull))

    #save feedbacks
    Tadd = '_Tfull' if Tfull else ''
    if full:
        np.save(f'./output/cmip_pic_feeds_binnedpic_full_mask_compare{Tadd}.npy',np.transpose(ys))
    else:
        np.save(f'./output/cmip_pic_feeds_binnedpic_2lims_{binpc[0]}{binpc[1]}_mask_compare{Tadd}.npy',[ys_4x,ys2_4x,ys3_4x])
        np.save(f'./output/cmip_pic_feeds_binnedpic_zero_mask_compare{Tadd}.npy',[ysz_4x,ysz2_4x,ysz3_4x])

def feedback_calc_B_obs(full=False,Tfull=False,err_type='cint'):
    '''
    Calculate reanalysis feedbacks.

    Input
    ---
    full      Bool; True if want to calculate full tropics, no regime
    Tfull     Bool; True to use global SSTA in regression, False to use tropical
    err_type  'cint' of 'se', either confidence error or standard error
    
    Output
    ---
    SWfeed_0,LWfeed_0,net_0,SWfeed_ncep,LWfeed_ncep,net_ncep    List of SW, LW and net feedbacks separated into regime using ERA5 and NCEP
    '''
    
    B=iris.load_cube('./input/B_era5.nc','b')
    B_ncep=iris.load_cube('./input/B_NCEP.nc')

    landsea_mask=iris.load_cube('./input/landsea_mask.nc')
    for i in ['longitude','latitude']:
        landsea_mask.coord(i).guess_bounds()
        try: B.coord(i).guess_bounds()
        except:pass
        try: B_ncep.coord(i).guess_bounds()
        except:pass

    t_cons=iris.Constraint(time = lambda t: datetime.datetime(2000,3,1)<=t.point<=datetime.datetime(2016,5,30))
    SWCRE=iris.load_cube("./input/SWCREadj_obs.nc",t_cons&iris.Constraint(latitude=lambda l:-30<l<30))
    LWCRE=iris.load_cube("./input/LWCREadj_obs.nc",t_cons&iris.Constraint(latitude=lambda l:-30<l<30))
    T_nc=iris.load_cube('./input/noaa_oi_sst_mnmean.nc',t_cons)

    #regridding (if required)
    landsea_mask=regrid(landsea_mask)
    landmask=np.where(landsea_mask.data<50,True,False)
    full_mask=regrid(landsea_mask,lat=90)
    landmask_full=np.where(full_mask.data<50,True,False)

    for i,nc in product(['longitude','latitude'],[SWCRE,LWCRE,T_nc]):
        try:nc.coord(i).guess_bounds()
        except:pass

    B_r=regrid(B)
    B_r_ncep=regrid(B_ncep)
    if Tfull:
        T = regrid(T_nc,lat=90)
    else:
        T=regrid(T_nc)
    SWCRE=regrid(SWCRE)
    LWCRE=regrid(LWCRE)

    #mask land
    n=np.shape(B)[0]
    B_r=B_r.copy(np.ma.masked_where([landmask]*n,B_r.data))
    B_r_ncep=B_r_ncep.copy(np.ma.masked_where([landmask]*n,B_r_ncep.data))
    if Tfull:
        T.data=np.ma.masked_where([landmask_full]*n,T.data)
    else:
        T.data=np.ma.masked_where(B_r.data.mask,T.data)
    SWCRE.data=np.ma.masked_where(B_r.data.mask,SWCRE.data)
    LWCRE.data=np.ma.masked_where(B_r.data.mask,LWCRE.data)

    B_r=B_r.extract(t_cons)
    B_r_ncep=B_r_ncep.extract(t_cons)

    #calculate percentiles
    fulldata=B_r.data.data[~B_r.data.mask]
    pcs=np.arange(5,100,5)
    Bbins=np.percentile(fulldata,[55,85])

    fulldata=B_r_ncep.data.data[~B_r_ncep.data.mask]
    pcs=np.arange(5,100,5)
    Bbins_ncep=np.percentile(fulldata,[55,85])
    
    #calculate anomalies
    wgts = iris.analysis.cartography.area_weights(T)
    Tavg=T.collapsed(['latitude', 'longitude'],iris.analysis.MEAN,weights=wgts)
    for nc in [T,Tavg,SWCRE,LWCRE]:
        try:
            icc.add_month_number(nc,'time')
        except:pass
    months=T.coord('month_number').points-1
    
    clims=[nc.aggregated_by('month_number',iris.analysis.MEAN) for nc in [Tavg,T,SWCRE,LWCRE]]
    clims_full=[np.array([nc.extract(iris.Constraint(month_number=m)).data for m in range(1,13)])[months] for nc in clims]
    anoms=[nc2-np.array(nc1) for nc1,nc2 in zip(clims_full,[Tavg,T,SWCRE,LWCRE])]
    Tavg = anoms[0]
    
    #feedback calculations
    if full:

        wgts = iris.analysis.cartography.area_weights(anoms[2])
        feedavgs = [nc.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=wgts).data 
            for nc in [anoms[2],anoms[3],anoms[2]+anoms[3]]]

        feeds_0=[[lin_reg(y,Tavg.data)['vals'][0],lin_reg(y,Tavg.data)[err_type][0]] for y in feedavgs]
        return(feeds_0)

    else:
        SWfeed_0=bin_by_B_regress2(anoms[2],B_r,anoms[1],bins=Bbins,lat=30,Tfull=Tfull,err_type=err_type)
        LWfeed_0=bin_by_B_regress2(anoms[3],B_r,anoms[1],bins=Bbins,lat=30,Tfull=Tfull,err_type=err_type)
        net_0=bin_by_B_regress2(anoms[2]+anoms[3],B_r,anoms[1],bins=Bbins,lat=30,Tfull=Tfull,err_type=err_type)

        SWfeed_ncep=bin_by_B_regress2(anoms[2],B_r_ncep,anoms[1],bins=Bbins_ncep,lat=30,Tfull=Tfull,err_type=err_type)
        LWfeed_ncep=bin_by_B_regress2(anoms[3],B_r_ncep,anoms[1],bins=Bbins_ncep,lat=30,Tfull=Tfull,err_type=err_type)
        net_ncep=bin_by_B_regress2(anoms[2]+anoms[3],B_r_ncep,anoms[1],bins=Bbins_ncep,lat=30,Tfull=Tfull,err_type=err_type)
        
        return(SWfeed_0,LWfeed_0,net_0,SWfeed_ncep,LWfeed_ncep,net_ncep)

if __name__ == '__main__':

    '''
    #calculating observed feedback and saving
    
    feedback_0 = np.array(feedback_calc_B_obs(err_type='se',Tfull=True,full=True))
    print(feedback_0)
    print(np.shape(feedback_0))

    das = [xr.DataArray(
        data=feedback_0[i,0],
        name = c+'_feed'
        )
        for i,c in enumerate(['sw','lw','net'])]
    das.extend([xr.DataArray(
        data=feedback_0[i,1],
        name = c+'_se'
        )
        for i,c in enumerate(['sw','lw','net'])])
    print(das)
    xr.merge(das).to_netcdf('./output/obs_pi_feeds_full_mask_compare_Tfull.nc')

    feedback_0 = np.array(feedback_calc_B_obs(err_type='se',Tfull=True))

    das = [xr.DataArray(
        data=[feedback_0[i,0],feedback_0[i+3,0]],
        dims=["obs_type","regime"],
        coords=dict(
            obs_type=['era5','ncep'],
            regime=['stable','mid','unstable']
        ),
        name = c+'_feed'
        )
        for i,c in enumerate(['sw','lw','net'])]
    das.extend(
        [xr.DataArray(
        data=[feedback_0[i,1],feedback_0[i+3,1]],
        dims=["obs_type","regime"],
        coords=dict(
            obs_type=['era5','ncep'],
            regime=['stable','mid','unstable']
        ),
        name = c+'_se'
        )
        for i,c in enumerate(['sw','lw','net'])]
    )
    xr.merge(das).to_netcdf('./output/obs_pi_feeds_2lims_5585_mask_compare_Tfull.nc')
    '''

