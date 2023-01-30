# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:29:19 2023
This module implements Cook's Distance as a metric for detecting outliers.
It's a bit redundant, so it's not included in __init__.
I might put it in, but IMO statistical testing is a better approach.
@author: ivanp
"""
import os
import time
import logging
from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy import stats

matplotlib.style.use("ggplot")

class CookDistance():
    """
    Implementation of Cooks Distance

    Source
    ------
    https://en.wikipedia.org/wiki/Cook%27s_distance
    """
    def __init__(self,df,x_var,y_var,bias=True):

        self.df = df.copy()
        self.y = self.df.pop(y_var).values.reshape(-1,1)

        self.x_name = x_var
        self.y_name = y_var

        if bias:
            self.df.insert(0,"bias",1)
        self.X = self.df.values

        #calculate distance dataframe
        self.dist_df = pd.DataFrame(self.dist_vector,index=self.df.index,columns=["CookD"])
        self.dist_df['OutlierThumb'] = False
        self.dist_df.loc[self.dist_df["CookD"] >= self.cvrt, "OutlierThumb"] = True

        self.dist_df['OutlierFdist'] = False
        self.dist_df.loc[self.dist_df["CookD"] >= self.cvfm, "OutlierFdist"] = True

    def draw_distribution(self,x_var=None,method="either"):
        """
        draws the distribution and notifies outliers
        """
        if x_var:
            x_values = self.X[:,x_var]
        else:
            if self.X.shape[1]>=2:
                x_values = self.X[:,1]
            else:
                x_values = self.X[:,0]

        draw_df = self.dist_df.copy()
        draw_df["y_pred"] = self.y_pred
        draw_df["y_true"] = self.y
        draw_df["x"] = x_values
        if method=="thumb":
            outlier_df = draw_df.loc[draw_df["OutlierThumb"]]
        elif method=="fdist":
            outlier_df = draw_df.loc[draw_df["OutlierFdist"]]
        elif method=="either":
            outlier_df = draw_df.loc[(draw_df["OutlierThumb"])|(draw_df["OutlierFdist"])]
        elif method=="both":
            outlier_df = draw_df.loc[(draw_df["OutlierThumb"])&(draw_df["OutlierFdist"])]
        else:
            outlier_df = draw_df.loc[draw_df["CookD"]>=method]


        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()
        colors = ["blue","red","green"]

        ax.plot(draw_df["x"],draw_df["y_pred"],color=colors[0])
        ax.vlines(draw_df["x"],draw_df["y_pred"],draw_df["y_true"],color=colors[1],lw=1e-1)
        if len(outlier_df)!=0:
            ax.scatter(outlier_df["x"],outlier_df["y_true"],color=colors[2],s=2)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(xlim[0],ylim[0]),max(xlim[1],ylim[1]))

        ax.text(x=0.72,y=0.65,s=f"MSE = {float(self.mean_squared_error):.4f}\n RSQ = {float(self.rsq):.4f}\n{len(outlier_df)} outliers\n{len(draw_df)} total\n{method=}",
                bbox=dict(facecolor='salmon', alpha=0.4),
                transform = ax.transAxes,
                fontsize=20)
        ax.set_xlabel(self.x_name,fontsize=20)
        ax.set_ylabel(self.y_name,fontsize=20)
        return fig, ax

    def get_outliers(self,method="either"):
        """
        There are two main ways to calculate cut-off for outliers:
            1. rule of thumb (method="thumb")
            The cut of value is 4 / n where n is the number of observations

            2. median of F distribution (method="fdist")
            The cut of value is the median (F0.5) of F distribution
            with df1 = p and df2 = n-p where p is the number of predictors
            and n is the number of observation

        Passing "either" applies OR criteria and passing "both" applies AND criteria
        """
        if method=="thumb":
            return self.dist_df.loc[self.dist_df["OutlierThumb"]]
        if method=="fdist":
            return self.dist_df.loc[self.dist_df["OutlierFdist"]]
        if method=="either":
            return self.dist_df.loc[(self.dist_df["OutlierThumb"])|(self.dist_df["OutlierFdist"])]
        if method=="both":
            return self.dist_df.loc[(self.dist_df["OutlierThumb"])&(self.dist_df["OutlierFdist"])]

        return self.dist_df.loc[self.dist_df["CookD"]>=method]

    @property
    def p(self):
        """
        The number of predictors
        First degree of freedom for F distribution
        """
        return np.linalg.matrix_rank(self.X)

    @property
    def n(self):
        """
        The number of observations
        To get second degree of freedom, subtract p from this
        """
        return self.X.shape[0]

    @property
    def hat_matrix(self):
        """
        Projection matrix of X
        """
        return self.X@(np.linalg.inv(self.X.T@self.X))@self.X.T

    @property
    def y_pred(self):
        """
        Predicted values of Y
        """
        return self.hat_matrix@self.y

    @property
    def e_vector(self):
        """
        Vector of residuals
        """
        #return self.y - self.hat_matrix@self.y
        return self.y - self.y_pred

    @property
    def leverages(self):
        """
        Diagonals of projection matrix
        """
        return np.diagonal(self.hat_matrix).reshape(-1,1)

    @property
    def mean_squared_error(self):
        """
        Exactly what it says
        """
        return (self.e_vector.T@self.e_vector)/(self.n-self.p)

    @property
    def rsq(self):
        """
        rsquared
        """
        return 1 - self.mean_squared_error/np.var(self.y)

    @property
    def dist_vector(self):
        """
        Vector containing Cook's Distance
        """
        dist_vector = (self.e_vector * self.e_vector * self.leverages)/((1-self.leverages)*(1-self.leverages))
        return dist_vector / (self.p * self.mean_squared_error)

    @property
    def cvrt(self):
        """
        critical value according to the rule of thumb
        """
        return 4 / self.n

    @property
    def cvfm(self):
        """
        critical value according to the median of F distribution
        """
        return stats.f.ppf(0.5,self.p,self.n-self.p)

def reset_loci(input_df,reset=False):
    """
    Resets loci if dataframe is indexed via locus.
    The example is chrom,start,end. This index splits
    the comma and returns them as columns named "chrom","start","end".
    If reset is passed as true, index is also reset.

    """
    df = input_df.copy()

    cols = list(input_df.columns)
    df[["chrom","start","end"]] = df.index.to_series().str.split(",",expand=True,regex=False)
    df = df[["chrom","start","end"]+cols]
    if reset:
        df.reset_index(inplace=True,drop=True)
    return df

def get_average_values(data_df,status_df,split_col="dwelling",sample_col="Sample Name",log2=False):
    """
    Exactly what it says
    """
    values = status_df[split_col].unique().tolist()
    average_df = pd.DataFrame(index=data_df.index)
    for value in values:
        samples = status_df[status_df[split_col]==value][sample_col]
        average_df[value] = data_df[samples].mean(axis=1)
        if log2:
            average_df[value] = np.log2(average_df[value])

    return average_df

def execute_operation(callable_func,iterable,multi_cores):
    """
    Executes callable function on elements of iterable
    using multiprocessing
    """
    _s = time.perf_counter()
    if multi_cores > 1:
        multi_cores = min(multi_cores,os.cpu_count())
        with Pool(multi_cores) as pool:
            _result = pool.imap_unordered(callable_func,iterable)
            result = [r for r in _result]
        _e = time.perf_counter()
    else:
        result = [callable_func(element) for element in iterable]
        _e = time.perf_counter()
    logging.info("Operation for %s done in %.4f seconds with %s total cores",len(iterable),_e-_s,multi_cores)
    return result


def _draw_cook_distance(element,status_df,x_var,y_var,bias,method,log2,save_prefix,
                        split_col,sample_col):

    name, dataframe = element
    average_values = get_average_values(dataframe,status_df,split_col,sample_col,log2)
    _s = time.perf_counter()
    cook = CookDistance(average_values,x_var,y_var,bias)
    fig, _ax = cook.draw_distribution(None,method=method)
    fig.suptitle(f"Linear regression of log2 average CN for {name}",fontsize=20,y=0.9)
    fig.savefig(f"{save_prefix}{name}.png",bbox_inches="tight")
    _e = time.perf_counter()
    #print(f"{name} of {len(dataframe)} done in {_e-_s:.4f} seconds",flush=True)
    logging.info("%s of %s size done in %.4f seconds",name,len(dataframe),_e-_s)
    plt.close(fig)

def draw_cook_dist(input_path,status_df,split_by_chrom=True,save_prefix="",x_var="surface",y_var="cave",
                   split_col="dwelling",sample_col="Sample Name",bias=True,log2=True,method="fdist",max_cores=None):
    """
    Draws linear regresion and outliers based on the Cook's distance.
    Relevant statistics (RSQ and MSE) are shown on graph.

    Parameters
    ----------
    input_path : str
        Path to data.
        Data must be numeric, with columns
        corresponding to "sample_col" column
        of "status_df" dataframe.
    status_df : pd.DataFrame
        Categorical dataframe that contains
        one column that has columns of numerical data outlined above,
        and one column "split_col" that divides the samples
        into two - one being "x_var" and other being "y_var"

    split_by_chrom : Bool, optional
        Whether to perform linear regression
        on the bulk or per chromosome. The latter
        is recommended for huge sizes.
        The default is True.

    save_prefix: str
        Where to save the visualizations

    split_col : str, optional
        Column in "status_df" that divides the samples
        into two - x_var and y_var.
        The default is "dwelling".

    x_var : str, optional
        Value in split_col.
        The default is "surface".
    y_var : str, optional
        The other value in "split_col".
        The default is "cave".

    sample_col : str, optional
        Column in "status_df" containing samples.
        This column must contain columns of numerical data.
        The default is "Sample Name".

    bias : Bool, optional
        Add a column of ones to the linear regression.
        So instead of Y = AX, have the model be
        Y = AX + B. The default is True.
    log2 : Bool, optional
        Use logarithm of two of values in data. The default is True.
    method : str, optional
        What kind of threshold value to use in
        determining outliers.
        For details see "get_outliers" method of CookDistance class.
        The default is "fdist".
    max_cores : int, optional
        Number of cores to use. The default is None.
    """
    #display locals
    for _k, _v in locals().items():
        logging.info("VARIABLE=%s, VALUE=%s",_k,_v)
        
    data = pd.read_csv(input_path,index_col=0)
    data = reset_loci(data)

    if split_by_chrom:
        chromosome_dataframes = list(tuple(data.groupby("chrom")))
    else:
        chromosome_dataframes = [("ALL_CHROMOSOMES",data)]

    if len(chromosome_dataframes)==1 or max_cores==1:
        max_cores=1

    call = partial(_draw_cook_distance,status_df=status_df,x_var=x_var,y_var=y_var,bias=bias,method=method,
                   log2=log2,save_prefix=save_prefix,split_col=split_col,sample_col=sample_col)
    logging.info("Drawing linear regression with %s cores and prefixing at %s",max_cores,save_prefix)
    execute_operation(call,chromosome_dataframes,max_cores)

def _get_cook_distance(element,status_df,x_var,y_var,bias,method,log2,
                        split_col,sample_col):
    """
    The function is the same as as the non-under version,
    but optimized to be frozen via functools partial.
    """
    name, dataframe = element
    average_values = get_average_values(dataframe,status_df,split_col,sample_col,log2)
    _s = time.perf_counter()
    df = CookDistance(average_values,x_var,y_var,bias).get_outliers(method)
    _e = time.perf_counter()
    #print(f"{name} of {len(dataframe)} done in {_e-_s:.4f} seconds",flush=True)
    logging.info("%s of %s size done in %.4f seconds",name,len(dataframe),_e-_s)
    return df

def get_cook_dist(input_path,status_df,split_by_chrom=True,x_var="surface",y_var="cave",
                   split_col="dwelling",sample_col="Sample Name",bias=True,log2=True,method="fdist",max_cores=None):
    """
    Gets outliers via Cook's distance regression model.

    Parameters
    ----------
    input_path : str
        Path to data.
        Data must be numeric, with columns
        corresponding to "sample_col" column
        of "status_df" dataframe.
    status_df : pd.DataFrame
        Categorical dataframe that contains
        one column that has columns of numerical data outlined above,
        and one column "split_col" that divides the samples
        into two - one being "x_var" and other being "y_var"

    split_by_chrom : Bool, optional
        Whether to perform linear regression
        on the bulk or per chromosome. The latter
        is recommended for huge sizes.
        The default is True.

    split_col : str, optional
        Column in "status_df" that divides the samples
        into two - x_var and y_var.
        The default is "dwelling".

    x_var : str, optional
        Value in split_col.
        The default is "surface".
    y_var : str, optional
        The other value in "split_col".
        The default is "cave".

    sample_col : str, optional
        Column in "status_df" containing samples.
        This column must contain columns of numerical data.
        The default is "Sample Name".

    bias : Bool, optional
        Add a column of ones to the linear regression.
        So instead of Y = AX, have the model be
        Y = AX + B. The default is True.
    log2 : Bool, optional
        Use logarithm of two of values in data. The default is True.
    method : str, optional
        What kind of threshold value to use in
        determining outliers. The default is "fdist".
    max_cores : int, optional
        Number of cores to use. The default is None.

    Returns
    -------
    pd.DataFrame
        Dataframe of outliers and their corresponding Cook Distance.

    """
    #display locals
    for _k, _v in locals().items():
        logging.info("VARIABLE=%s, VALUE=%s",_k,_v)
        
    data = pd.read_csv(input_path,index_col=0)
    data = reset_loci(data)

    if split_by_chrom:
        chromosome_dataframes = list(tuple(data.groupby("chrom")))
    else:
        chromosome_dataframes = [("ALL_CHROMOSOMES",data)]

    if len(chromosome_dataframes)==1 or max_cores==1:
        max_cores=1

    call = partial(_get_cook_distance,status_df=status_df,x_var=x_var,y_var=y_var,bias=bias,method=method,
                   log2=log2,split_col=split_col,sample_col=sample_col)
    logging.info("Calculating Cook's distance and returning outliers ")
    outliers_df = execute_operation(call,chromosome_dataframes,max_cores)

    return pd.concat(outliers_df,axis=0)