import iris,glob
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
from iris.util import equalise_attributes

import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import interp1d
import iris.coord_categorisation as icc
import pandas as pd
import xarray as xr
import intake
import seaborn as sns
import datetime
import pandas as pd
from scipy.stats import sem
from dateutil.relativedelta import relativedelta

def binning(cl_mod,B_mod,T_mod,Tfull=False):
    '''
    bin cloud amount and cloud response to SST by B

    Inputs
    ---
    cl_mod:      cloud fraction in isccp format (xarray)
    B_mod:       buoyancy (xarray)
    T_mod:       SST (xarray)
    Tfull:       True: use global SST in regression
                 False: use subset of SST in regression
    '''
    #prepare data
    cl_nc=cl_mod.sum('tau')
    cl_nc = cl_nc.sel(latitude=slice(-30,30))

    high_cl=cl_nc[:,:,:,:2].sum('plev7')
    low_cl=cl_nc[:,:,:,-2:].sum('plev7')
    mid_cl=cl_nc[:,:,:,2:-2].sum('plev7')

    #create monthly anomalies
    Tclim = T_mod.groupby('time.month').mean('time')
    Tanom = T_mod.groupby('time.month') - Tclim
    highclim = high_cl.groupby('time.month').mean('time')
    highcl_anom = high_cl.groupby('time.month') - highclim
    lowclim = low_cl.groupby('time.month').mean('time')
    lowcl_anom = low_cl.groupby('time.month') - lowclim

    #take points over ocean in tropics and global and get Bbins
    B_sub = B_mod.extract(iris.Constraint(latitude=lambda l: -30<=l<=30))
    fulldata=B_sub.data.data[~B_sub.data.mask]
    pcs=np.arange(5,100,5)
    Bbins=np.percentile(fulldata,pcs)

    #feedbacks in each bin
    ys = []
    for nc in highcl_anom,lowcl_anom:
        ys.extend(bin_by_B_regress(nc.to_iris(),B_sub,Tanom.to_iris(),bins=Bbins,lat=30,Tfull=Tfull))
    
    df_cloud = pd.DataFrame(np.transpose(ys),columns = ['high_feed','high_cint','low_feed','low_cint'])

    #amount in each bin
    df=pd.DataFrame(np.transpose([high_cl.values[~B_sub.data.mask],low_cl.values[~B_sub.data.mask],
                                  mid_cl.values[~B_sub.data.mask],
                fulldata,pd.qcut(fulldata,20,labels=False),
                                 pd.cut(fulldata,20,labels=False)]),
                columns=['highcl','lowcl','midcl','B','Bbin','Bbinab'])
    
    return(df.groupby('Bbin').agg(['mean','sem','std']),df.groupby('Bbinab').agg(['mean','sem','std']),df_cloud)

def sub_nc(nc,Bbool):
    '''
    average of nc where Bbool is True
    '''
    for i in ['longitude','latitude']:
        try: nc.coord(i).guess_bounds()
        except:pass
    
    wgts = iris.analysis.cartography.area_weights(nc)
    nc=nc.copy(np.ma.masked_where(~Bbool,nc.data))
    return(nc.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=wgts).data)

def lin_reg(y,x,c_give=False):
    '''
    linear regression wrapper function
    '''
    x=np.ravel(x);np.ravel(y)

    if len(x)==0:
        return([np.nan])
    x=sm.add_constant(x)
    fit=sm.OLS(y,x)
    cint=fit.fit().conf_int(alpha=0.05)

    res = fit.fit()
    if len(res.params)==2:
        intercept=res.params[0]
        slope=res.params[1]
        if c_give: return slope-cint[1][0],intercept-cint[0][0]
        return slope,intercept
    else:
        if c_give: return [res.params[0]-cint[0][0]]
        return res.params
    
def bin_by_B_regress(feed,B,T,bins,lat=80,Tfull=False):
    '''
    regression in bins of B

    Input
    ---
    feed       values to be regressed (xarray, time,lat,lon)
    B          buoyancy (xarray, time,lat,lon)
    T          SST (xarray, time,lat,lon)
    bins       bin edges for binning (list)
    lat        latitude range to do calculation (int)
    Tfull      bool for global SST use

    Output
    ---
    feed_avg       list of feedbacks over bins
    feed_cint      list of confidence interval for regression over bins
    '''
    
    for i in ['longitude','latitude']:
        try: T.coord(i).guess_bounds()
        except:pass
    
    B=B.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))
    feed=feed.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))
    if not Tfull:
        T=T.extract(iris.Constraint(latitude=lambda l: -1*lat<=l<=lat))

    wgts = iris.analysis.cartography.area_weights(T)
    Tavg=T.collapsed(['latitude','longitude'],iris.analysis.MEAN,weights=wgts).data
    feed_avg=[lin_reg(sub_nc(feed,np.ma.filled(np.ma.less(B.data,bins[0]),False)),Tavg)[0]]
    feed_cint=[lin_reg(sub_nc(feed,np.ma.filled(np.ma.less(B.data,bins[0]),False)),Tavg,c_give=True)[0]]
    
    limcheck=lambda a,b: np.ma.filled(np.ma.logical_and(np.ma.greater_equal(B.data,a),np.ma.less(B.data,b)),False)
    feed_avg.extend(map(lambda lim1,lim2: lin_reg(sub_nc(feed,limcheck(lim1,lim2)),Tavg)[0],bins[:-1],bins[1:]))
    feed_cint.extend(map(lambda lim1,lim2: lin_reg(sub_nc(feed,limcheck(lim1,lim2)),Tavg,c_give=True)[0],bins[:-1],bins[1:]))
    feed_avg.extend([lin_reg(sub_nc(feed,np.ma.filled(np.ma.greater_equal(B.data,bins[-1]),False)),Tavg)[0]])
    feed_cint.extend([lin_reg(sub_nc(feed,np.ma.filled(np.ma.greater_equal(B.data,bins[-1]),False)),Tavg,c_give=True)[0]])
    
    return(feed_avg,feed_cint)

def regrid(nc,res=1,lat=30):
    '''
    regrid function using iris
    '''
    for i in ['longitude','latitude']:
        nc.coord(i).units='degrees'
        try: nc.coord(i).guess_bounds()
        except:pass
    latitude = iris.coords.DimCoord(np.arange(-lat, lat+res, res),
                     standard_name='latitude',
                     units='degrees')
    latitude.guess_bounds()
    xlen=len(np.arange(-lat, lat+res, res))

    longitude = iris.coords.DimCoord(np.arange(0, 360, res),
                         standard_name='longitude',
                         units='degrees')
    longitude.guess_bounds()
    ylen = len(np.arange(0, 360, res))

    newgrid = iris.cube.Cube(np.zeros((xlen, ylen), np.float32),
        dim_coords_and_dims=[(latitude, 0),(longitude, 1)])

    nc_r=nc.regrid(newgrid,iris.analysis.AreaWeighted(mdtol=0.5))
    return(nc_r)

def binning_cre(cre_mod,B_mod,T_mod,Tfull=False):
    '''
    cloud feedback (SW and LW) in B bins
    '''
    Tclim = T_mod.groupby('time.month').mean('time')
    Tanom = T_mod.groupby('time.month') - Tclim
    B_mod = B_mod.extract(iris.Constraint(latitude=lambda l: -30<=l<=30))
    
    fulldata=B_mod.data.data[~B_mod.data.mask]

    pcs=np.arange(5,100,5)
    Bbins=np.percentile(fulldata,pcs)
    ys = []
    for nc in cre_mod:
        ys.extend(bin_by_B_regress(nc.to_iris(),B_mod,Tanom.to_iris(),bins=Bbins,lat=30,Tfull=Tfull))
        
    df_cloud = pd.DataFrame(np.transpose(ys),columns = ['SW_feed','SW_cint','LW_feed','LW_cint','net_feed','net_cint'])
    
    return(df_cloud)

if __name__=="__main__":

    #define constant parameters
    Tfull=True
    latmax = 30 if not Tfull else 90
    
    #load files
    isccp_nc=iris.load_cube('./input/clisccp_198307-200806.nc','isccp_cloud_area_fraction')
    starttime=datetime.datetime(1983,6,15)
    dates=[starttime+relativedelta(months=i) for i in isccp_nc.coord('time').points]
    newtpoints=[(i-starttime).days for i in dates]
    isccp_nc.coord('time').points=newtpoints
    isccp_nc.coord('time').units='days since 1983-6-15'
    
    t_cons=iris.Constraint(time = lambda t: datetime.datetime(2000,3,1)<=t.point<=datetime.datetime(2016,5,30))
    
    #all data can be retrieved online or calculated using provided scripts
    
    B = iris.load_cube('./input/B_era5.nc','b'&t_cons)
    landsea_mask = iris.load_cube('./input/landsea_mask.nc')
    T_nc = iris.load_cube('./input/noaa_oi_sst_mnmean.nc')
    w500_e5 = iris.load_cube("./input/era5_w500_2000-2016.nc")
    SWCRE = iris.load_cube("./input/SWCREadj_obs.nc",t_cons)
    LWCRE = iris.load_cube("./input/LWCREadj_obs.nc",t_cons)
    cre_ncs = iris.load('./input/CERES*.nc',t_cons)
    names = ['toa_sw_clr_c_mon','toa_lw_clr_c_mon','toa_sw_all_mon','toa_lw_all_mon']
    SW_clr,LW_clr,SW,LW = map(lambda n: cre_ncs.extract(iris.Constraint(cube_func=lambda c: c.var_name==n))[0],names)
    SWCRE_raw = SW_clr-SW
    LWCRE_raw = LW_clr-LW
    
    #regridding
    landsea_mask=regrid(landsea_mask,res=2,lat=latmax)
    landmask=np.where(landsea_mask.data<50,True,False)
    
    B_r,T_r,w500_r,cl_nc,SWCRE,LWCRE,SWCRE_raw,LWCRE_raw = list(map(lambda nc: regrid(nc,res=2,lat=latmax),
        [B,T_nc,w500_e5,isccp_nc,SWCRE,LWCRE,SWCRE_raw,LWCRE_raw]))
    cl_nc.coord('longitude').circular=True

    #masking land
    n=np.shape(B)[0]
    B_r.data=np.ma.masked_where([landmask]*n,B_r.data)
    
    nw=np.shape(w500_r)[0]
    w500_r.data=np.ma.masked_where([landmask]*nw,w500_r.data)

    #make data consistent time and correct dimensions and format
    
    cl_nc.transpose([0,3,4,1,2])
    print(cl_nc)
    startdate=datetime.datetime(2000,3,1)
    enddate=datetime.datetime(2008,6,30)
    t_cons=iris.Constraint(time=lambda t: startdate<=t.point<=enddate)
    B_mod,w500_mod,T_mod,cl_mod = [nc.extract(t_cons) for nc in [B_r,w500_r,T_r,cl_nc]]
    
    T_mod.data=np.ma.masked_where([landmask]*np.shape(T_mod)[0],T_mod.data)
    cl_mod=xr.DataArray.from_iris(cl_mod)
    T_mod=xr.DataArray.from_iris(T_mod)

    #binning calculations
    
    df_agg_B,df_agg_Bab,df_cloud=binning(cl_mod,B_mod,T_mod,Tfull=Tfull)
    df_agg_w,df_agg_wab,df_cloud_w=binning(cl_mod,-1*w500_mod,T_mod,Tfull=Tfull)
    df_agg_w.index=np.arange(2.5,100,5)
    zero_pc=interp1d(df_agg_w.B['mean'].values,df_agg_w.B.index.values)(0)
    
    B_sub,SWCRE_sub,LWCRE_sub = [nc.extract(iris.Constraint(latitude = lambda l: -30<=l<=30))
        for nc in [B_r,SWCRE_raw,LWCRE_raw]]
    B_data=B_sub.data.data[~B_sub.data.mask]
    SWdata=SWCRE_sub.data.data[~B_sub.data.mask]
    LWdata=LWCRE_sub.data.data[~B_sub.data.mask]
    
    df=pd.DataFrame(np.transpose([SWdata,LWdata,pd.qcut(B_data,20,labels=False)]),columns=['SW','LW','bin'])
    df_grouped=df.groupby('bin').agg([np.mean,sem,'std'])
    df_grouped['SW']['mean'] *= -1
    df_grouped.index=np.arange(2.5,100,5)
    
    B_mod=B_r.extract(t_cons2)
    T_mod=T_r.extract(t_cons2)
    T_mod.data=np.ma.masked_where(B_mod.data.mask,T_mod.data)
    
    ncs_mod=[SWCRE,LWCRE]
    ncs_mod=[xr.DataArray.from_iris(nc) for nc in ncs_mod]
    T_mod=xr.DataArray.from_iris(T_mod)
    df_cre = binning_cre(ncs_mod+[ncs_mod[0]+ncs_mod[1]],B_mod,T_mod,Tfull=Tfull)
    df_cre.index=np.arange(2.5,100,5)
    
    #=======================================================================
    #PLOTTING
    #=======================================================================
    
    #Fig. S3
    #BIn by w500 figure
    
    sns.set_style("whitegrid")
    fig,axs=plt.subplots(nrows=2,figsize=[8,8])
    ax=axs[0]

    df_agg_w.highcl.plot(y='mean',yerr='std',capsize=4,ax=ax,label='deep clisccp (<310 hPa)',color='b',linestyle='dashed')
    df_agg_w.lowcl.plot(y='mean',yerr='std',capsize=4,ax=ax,color='r',label='shallow clisccp (>680 hPa)')
    ax.set_ylim([0,80])
    #ax.grid()
    ax.set_xlabel(r'$-1*\omega_{500}$ pc')
    ax.set_ylabel('%')
    ax.legend(loc='upper left')
    ax.axvline(zero_pc,color='k',linestyle='dashed'),
    ax.text(zero_pc,ax.get_ylim()[1]+2,r'ERA5 $\omega_{500}$=0',ha='center',va='center')
    
    ax=axs[1]
    df_cloud_w.index=np.arange(2.5,100,5)
    
    df_cloud_w.plot(y='high_feed',yerr='high_cint',capsize=4,ax=ax,label='deep clisccp (<310 hPa)',color='b',linestyle='dashed',legend=False)
    df_cloud_w.plot(y='low_feed',yerr='low_cint',capsize=4,ax=ax,color='r',label='shallow clisccp (>680 hPa)',legend=False)
    
    ax.set_ylim([-20,20])
    ax.set_xlabel(r'$-1*\omega_{500}$ pc')
    ax.set_ylabel(r'% K$^{-1}$')
    fig.tight_layout()
    ax.axvline(zero_pc,color='k',linestyle='dashed')
    [ax.axvline(x,color='k',linestyle='dashdot') for x in [55,85]]
    ax.set_xticks([55, 85], minor=True)
    ax.set_xticklabels([55, 85], minor=True)
    [ax.set_title(t,loc='left') for ax,t in zip(axs,['a) cloud fraction','b) cloud response to SSTA'])]
    fig.tight_layout()
    Tadd = '_Tfull' if Tfull else ''
    fig.savefig(f'./paper_figs/supp/FigS3.png',dpi=300,bbox_inches='tight',pad_inches=.1)

    #===================================================================
    #Fig. 2
    #Binning CRE and cloud amount by B
    
    df_agg_B.index=np.arange(2.5,100,5)
    zero_pc=interp1d(df_agg_B.B['mean'].values,df_agg_B.B.index.values)(0)

    fig,axs=plt.subplots(nrows=2,ncols=2,figsize=[12,8])
    ax=axs[0][0]
    df_agg_B.highcl.plot(y='mean',yerr='std',capsize=4,ax=ax,label='deep clisccp (<310 hPa)',color='b',linestyle='dashed')
    df_agg_B.lowcl.plot(y='mean',yerr='std',capsize=4,ax=ax,color='r',label='shallow clisccp (>680 hPa)')
    ax.set_ylim([0,80])
    #ax.grid()
    ax.set_ylabel('%')
    ax.legend(loc='upper left')
    ax.axvline(zero_pc,color='k',linestyle='dashed'),
    ax.text(zero_pc,ax.get_ylim()[1]+2,'ERA5 B=0',ha='center',va='center')
    
    ax=axs[0][1]
    df_cloud.index=np.arange(2.5,100,5)
    
    df_cloud.plot(y='high_feed',yerr='high_cint',capsize=4,ax=ax,label='deep clisccp (<310 hPa)',color='b',linestyle='dashed')
    df_cloud.plot(y='low_feed',yerr='low_cint',capsize=4,ax=ax,color='r',label='shallow clisccp (>680 hPa)')
    
    ax.set_ylim([-20,20])
    #ax.grid()
    ax.set_ylabel(r'% K$^{-1}$')
    fig.tight_layout()
    #plt.legend(loc='upper left')
    ax.axvline(zero_pc,color='k',linestyle='dashed')
    [ax.axvline(x,color='k',linestyle='dashdot') for x in [55,85]]
    ax.set_xticks([55, 85], minor=True)
    ax.set_xticklabels([55, 85], minor=True)
    [ax.set_title(t,loc='left') for ax,t in zip(axs[0],['a) cloud fraction','b) cloud fraction response to SSTA'])]
    
    ax=axs[1][0]
    
    df_grouped['SW'].plot(y='mean',yerr='std',capsize=2,ax=ax,grid=True,label='-SWCRE',color='b',linestyle='dashed')
    df_grouped['LW'].plot(y='mean',yerr='std',capsize=2,ax=ax,grid=True,label='LWCRE',color='r')
    ax.axvline(zero_pc,color='k',linestyle='dashed')
    ax.set_ylabel(r'W m$^{-2}$')
    
    ax=axs[1][1]
    df_cre.plot(y='SW_feed',yerr='SW_cint',capsize=2,ax=ax,grid=True,label='SW cloud feed',color='b',linestyle='dashed')
    df_cre.plot(y='LW_feed',yerr='LW_cint',capsize=2,ax=ax,grid=True,label='LW cloud feed',color='r')
    df_cre.plot(y='net_feed',yerr='net_cint',capsize=2,ax=ax,grid=True,label='net cloud feed',color='k',linewidth=1)
    
    ax.axvline(zero_pc,color='k',linestyle='dashed')
    [ax.axvline(x,color='k',linestyle='dashdot') for x in [55,85]]
    ax.set_ylabel(r'W m$^{-2}$ K$^{-1}$')
    [ax.set_xlabel('B pc') for ax in np.ravel(axs)]
    [ax.set_title(t,loc='left') for ax,t in zip(axs[1],['c) CRE','d) Interannual cloud feedback'])]
    ax.set_xticks([55, 85], minor=True)
    ax.set_xticklabels([55, 85], minor=True)
    [ax.text(zero_pc,ax.get_ylim()[1]+2,'ERA5 B=0',ha='center',va='center') for ax in np.ravel(axs)]
    [ax.legend(loc='upper left') for ax in np.ravel(axs)]
    
    fig.tight_layout()
    
    fig.savefig(f'./paper_figs/Fig2.png',dpi=300,bbox_inches='tight',pad_inches=.1)
