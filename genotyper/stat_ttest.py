# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:05:06 2023
Better statistical testing.

The module implements Welch's T-Test for large amounts of data stored in matrix.
Two things are neccessary data-inputs - numeric dataframe and categorical dataframe:

===============================================================================
    NUMERIC DATAFRAME
===============================================================================
The input numeric data should be stored in all-numerical dataframe of M x N size.
The elements of group that is tested between and/or within are columns of the dataframe
whilst the independent features are stored in indices of numerical dataframe.

An example of a numerical dataframe:

>>>        Choy01  Choy05  Choy06  Choy09  ...  Rascon13  Rascon15  Rascon8  Rascon6
>>> 0      2.5450   1.874  3.3480   2.668  ...    2.6660    2.7520   2.5920   2.5860
>>> 1      1.7380   1.686  1.7360   1.814  ...    1.6580    1.8030   1.6910   1.7830
>>> 2      1.8230   1.879  1.8450   1.978  ...    1.4480    1.4580   1.4430   1.3630
>>> 3      1.4260   1.207  1.6900   1.663  ...    1.4375    1.5510   1.8170   1.4070
>>> 4      1.4375   1.459  1.1360   1.552  ...    1.7720    1.5780   1.4780   1.7480
>>> ...     ...     ...     ...     ...  ...       ...       ...      ...      ...
>>> 15053  1.5860   1.341  1.9990   1.323  ...    2.0940    1.8390   1.9740   1.6790
>>> 15054  4.4100   4.277  1.9960   3.602  ...    0.3586    0.3025   0.3928   0.3416
>>> 15055  2.1880   2.480  2.0330   1.639  ...    1.7870    1.7350   1.8300   1.5830
>>> 15056  0.6567   0.605  0.5923   0.647  ...    0.7114    0.6600   0.5537   0.6390
>>> 15057  2.3630   1.506  2.5410   2.246  ...    2.8950    2.2900   2.6400   2.2910

Every row is an independent value of something, while every column is a particular sample
that might belong to a certain group.

===============================================================================
    CATEGORICAL DATAFRAME
===============================================================================

The categorical dataframe should be of N x K size, and should contain one column
that is related to columns of numerical dataframe. Other columns should be grouping
variables for statistical testing.

An example of categorical dataframe:

>>>            dwelling lineage    region
>>> Sample Name
>>> Choy01       surface     new  Rio Choy
>>> Choy05       surface     new  Rio Choy
>>> Choy06       surface     new  Rio Choy
>>> Choy09       surface     new  Rio Choy
>>> ...           ...        ...      ...
>>> Rascon13     surface     old    Rascon
>>> Rascon15     surface     old    Rascon
>>> Rascon8      surface     old    Rascon
>>> Rascon6      surface     old    Rascon


As you can see, values in index are all columns in numerical dataframe.

===============================================================================
    STATISTICAL TESTING
===============================================================================

After both dataframes are placed, functions "stat_test" and "stat_test_wrangler"
are both used to perform statistical testing. The latter calls the former multiple
times.

There are two parameters used for testing - "between" and "within". Both
parameters must be valid columns in categorical dataframe, with "within" being optional.

For example, specifying "between" to be "dwelling" will separate columns
in "surface" and "cave" and perform statistical testing based on that.

Further specifying "within" to be "region" will perform testing between
all "cave" regions and "surface" regions.

Alternatively, if the columns of numeric dataframe and not find in
indices of categorical dataframe then you can specify the column of categorical
dataframe they are in via "sample" parameter.

===============================================================================
    STATISTICAL PARAMETERS
===============================================================================

Additional parameters are used to to tweak statistical testing.

### P-value adjusting

It's expected than many of Welch T-Tests are performed, so P-values should be
adjusted per column. Methods that are used to adjust P-values are:
    FWER : "bonf","holm","sidak"
    FDR : "fdr_bh", "fdr_by"
Details about these methods are found in pingouin module. If "within" parameter
is not specified, the result of statistical testing is a single column. However, if
it is specified, the result is a dataframe of N x P dimensions.

P values are adjusted column-wise and row-wise via "row_method","col_method".
Which adjustmend is performed first is controlled with "adjust_first" method.

Specify threshold of significance via "alpha" parameter, e.g. "alpha=0.05"

### Effect size

There are two main methods specified for effect size and both are controlled
via "effsize" parameter. The first is Cohen's D ("D") and the second one
is a logarithm base two of ratio of means ("FC").

===============================================================================
    FURTHER DETAILS
===============================================================================

Function "stat_test_wrangler" is made to ease the syntax of multiple testing for different
parameters.
It takes two strict paramters, "data_df" and "status_df", and then it takes
any amount of strictly positional arguments, and then it takes strictly keyword arguments.

The strictly positional arguments are either a tuple or a dictionary pairs
passed to "between" and "within".

Example:
    >>> stat_elements = [("dwelling","region"),("dwelling",None)]
    OR
    >>> stat_elements = [{"between":"dwelling","within":"region"},
    >>>                     {"between":"dwelling","within":None}]

Strictly keyword arguments are all parameters related to statistical testing.


===============================================================================
    EXAMPLE CODE
===============================================================================



>>> data = pd.read_csv("genotyper_data/genotyped_tested_dwelling.csv",index_col=0)
>>> status = pd.read_csv("linker.csv")[["Sample Name","region","dwelling","lineage"]]
>>> status = status.set_index("Sample Name")

>>> stat_elements = [("dwelling","region"),
               ("dwelling",None)]
OR
>>> stat_elements = [{"between":"dwelling","within":"region"},
>>>                  {"between":"dwelling","within":None}]

>>> stat_params = {"adjust_first":"row",
>>>                "row_method":"holm",
>>>                "col_method":"fdr_bh",
>>>                "alpha":0.05,
>>>                "effsize":"D"}


>>> results = stat_test_wrangler(data,status,*stat_elements,**stat_params)

@author: ivanp
"""


import itertools
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

def generate_iterative_columns(status_df,between,within=None,sample=None):
    """
    From a given categorical dataframe, generates
    different combinations of columns.

    Example:
        We have a dataframe where column "Sample" represents
        a samples, where column "Tumor" is a binary value between "tumor" or "healthy",
        and where column "Tissue" has values such as "Liver","Spleen","Kidney".

        Setting "sample='Sample'" and "between='Tumor'" will return the following:

        >>> {"Tumor:Yes_No" : (["TL1","TL2","TS1","TS2"...],["HL1","HL2"...]) }

        Setting "within='Tissue'" will return the following:

        >>> {"Liver(tumor)-Liver(healthy)" : (["TL1","TL2"],["HL1","HL2"])
             "Kidney(tumor)-Kidney(tumor)" : (["TK1","TK2"],["HK1","HK2"]),
             ....}


    Parameters
    ----------
    status_df : pandas dataframe
        A categorical dataframe containing information about samples.
    between : str
        Column in status_df.
    within : str, optional
        Column in status_df. If specified, will generate multiple values.
    sample : str, optional
        Column in status_df. If not specified, assumption will be made
        that index contains sample names

    Returns
    -------
    dict where keys correspond to names,
    and values are a 2-ple of samples
    """

    resulting_dictionary = dict()

    if sample is None:
        sample = status_df.index.name
        status_df = status_df.reset_index()

    if within is None:
        grouper = status_df.groupby([between])[sample].unique().to_dict()
        key_1, key_2 = grouper.keys()
        x_cols, y_cols = grouper[key_1].tolist(),grouper[key_2].tolist()
        return { f"{between}:{key_1}_{key_2}" : (x_cols,y_cols)}


    grouper = status_df.groupby([between])[within].unique().to_dict()
    key_1, key_2 = grouper.keys()
    values_1, values_2 = grouper[key_1], grouper[key_2]

    for value_1, value_2 in itertools.product(values_1,values_2):
        name = f"{value_1}({key_1})-{value_2}({key_2})".replace(" ","_")
        x_cols = status_df[status_df[within]==value_1][sample].tolist()
        y_cols = status_df[status_df[within]==value_2][sample].tolist()
        resulting_dictionary[name]=(x_cols,y_cols)
    return resulting_dictionary

def _welch_ttest_scipy(df,x_cols,y_cols):
    """
    Welch t-test optimized for pandas.DataFrame.apply function

    This is the Backup version. Sometimes, pingouin module will
    not act nicely, and will give out TypeError: len() of unsized object.

    I have no idea why this happens. The duct-tape solution is
    to jerry-rig scipy.stats module version.

    This backup is more complicated - stats.ttest_ind
    will take two arrays, and ttest_1samp will take one array and one int/float.

    Parameters
    ----------
    df : DataFrame.
    x_cols : columns to split values into one array.
    y_cols : columns to split values into another array.

    Returns
    -------
    P value of two sided Welch T-Test
    """
    def is_unique(iterable):
        """"Check if iterable has only one unique value"""
        return len(set(iterable))==1

    x = df[x_cols].tolist()
    y = df[y_cols].tolist()

    if not is_unique(x) and not is_unique(y):
        pval = stats.ttest_ind(x,y)[1]
    if is_unique(x) and is_unique(y):
        pval = np.nan
    if is_unique(x) and not is_unique(y):
        popmean = x[0]
        pval = stats.ttest_1samp(y,popmean)[1]
    if is_unique(y) and not is_unique(x):
        popmean = y[0]
        pval = stats.ttest_1samp(x,popmean)[1]
    return pval

def row_wise_adjustment(pvalue_df,method):
    """Adjust pvalues contained in ROWS"""
    result = pd.DataFrame(index=pvalue_df.index)
    columns = pvalue_df.columns.tolist()
    if pvalue_df.shape[1]==1:
        columns = columns[0]
    result[columns] = pvalue_df.apply(lambda row: pg.multicomp(row.tolist(),method=method)[1]
                                      ,axis=1,result_type="expand")
    return result

def col_wise_adjustment(pvalue_df,method):
    """Adjust pvalues contained in COLUMNS"""
    result = pd.DataFrame(index=pvalue_df.index)
    for col in pvalue_df.columns:
        result[col] = pg.multicomp(pvalue_df[col].tolist(),method=method)[1]

    return result

def _compute_cohen(df,x_cols,y_cols):
    """
    Calculates Cohen's D as effect size metric

    Parameters
    ----------
    df : DataFrame.
    x_cols : columns to split values into one array.
    y_cols : columns to split values into another array.

    Returns
    -------
    Cohen's D
    """
    x = df[x_cols].tolist()
    y = df[y_cols].tolist()
    return pg.compute_effsize(x,y)

def _compute_logfold(df,x_cols,y_cols):
    """
    Calculates log(base2) of ratio of means as effect size metric

    Parameters
    ----------
    df : DataFrame.
    x_cols : columns to split values into one array.
    y_cols : columns to split values into another array.

    Returns
    -------
    LogFoldChange of two means
    """
    x = df[x_cols].mean()
    y = df[y_cols].mean()

    if y==0 and x==0:
        return 0
    if y==0:
        return np.log2(x)
    if x==0:
        return -np.log2(y)

    return np.log2(x/y)

def stat_test(data_df,status_df,between,within=None,sample=None,
              adjust_first="row",row_method="holm",col_method="fdr_bh",
              alpha=0.05,effsize="D"):
    """
    Performs statistical testing on a given numerical dataframe
    outlined in "data_df", based on categorical values based on values
    in "status_df".

    Parameter "sample" denotes where sample column is found in "status_df"
    (if not specified, index is assumed to be sample column).

    Parameter "between" denotes what column in "status_df" contains binary
    values that a Welch T-Test is between (e.g. a column "Tumor" can contain
    values like "Healthy" and "Cancer").

    Parameter "within" denotes additional level of statistical testing.

    Parameter "effsize" denotes what kind of effect size will be reported.
    By default, "D" is for Cohen's D and "FC" is for log2 fold change of means.

    Alpha is level of statistical significance.

    Depending on parameter within you can end up with multidimensional P value dataframe.
    Pvalues are then adjusted column-wise and row-wise.

    Parameter "adjust_first" denotes what is adjusted first, "row" or "col".
    Paramters "row_method" and "col_method" are used to set what P value adjustment method is used.
    Valid parameters are found in pingouin documentation :
        FWER : "holm","sidak","bonf"
        FDR : "fdr_bh","fdr_by"

    """

    #prepare two dataframes
    # one for ovalues
    # one for effect size
    pvalue_df = pd.DataFrame(index=data_df.index)
    eff_df = pd.DataFrame(index=data_df.index)


    #check for effect size
    if effsize=="D":
        eff_func = _compute_cohen
    elif effsize=="FC":
        eff_func = _compute_logfold
    else:
        eff_df = None

    #generate test information
    #this is a dictionary of "name" : (x_cols,y_cols)
    test_info = generate_iterative_columns(status_df,between,within,sample)

    for d,(x_cols,y_cols) in test_info.items():
        pvalue_df[f"FDR_{d}"] = data_df.apply(_welch_ttest_scipy,axis=1,x_cols=x_cols,y_cols=y_cols)
        if eff_df is not None:
            eff_df[f"{effsize}_{d}"] = data_df.apply(eff_func,axis=1,x_cols=x_cols,y_cols=y_cols)
    #adjust for statistical testing - control order
    if adjust_first == "row":
        pvalue_adjusted = row_wise_adjustment(pvalue_df,row_method)
        pvalue_adjusted = col_wise_adjustment(pvalue_adjusted,col_method)
    else:
        pvalue_adjusted = col_wise_adjustment(pvalue_df,col_method)
        pvalue_adjusted = row_wise_adjustment(pvalue_adjusted,row_method)

    #after adjusting -> report significance based on alpha
    significant = pd.DataFrame(data=np.where(pvalue_adjusted <= alpha,True,False),
                                       index=pvalue_adjusted.index,columns=pvalue_adjusted.columns)
    #merge them
    result = pd.merge(pvalue_adjusted,significant,left_index=True,right_index=True,
                      suffixes=('_pvalues', f'<{alpha}'))
    if eff_df is None:
        return result
    return pd.merge(result,eff_df,left_index=True,right_index=True)

def stat_test_wrangler(data_df,status_df,*args,**kwargs):
    """
    Streamlines statistical testing.
    "data_df" is a numerical pandas dataframe,
    "status_df" is a categorical pandas dataframe, and those two
    are required arguments.

    Positional arguments after those two are used to
    create multiple dataframes from testing.

    Keyword arguments after that are passed to "stat_test" function.


    Example usage:

        >>> mydata = pd.read_csv("mydata.csv")
        >>> mysamples = pd.read_csv("mysamples.csv")
        >>> stat_elements = [("tumor","tissue"),
        >>>                  ("tumor",None)]
        >>> stat_params = {"adjust_first":"col","effsize":"FC","alpha":0.01}

        >>> result = stat_test_wrangler(mydata,mysamples,*stat_elements,**stat_params)

    """

    result_df = [data_df.copy()]

    sample = kwargs.pop("sample",None)

    for stat_element in args:
        if isinstance(stat_element,dict):
            between = stat_element["between"]
            within = stat_element["within"]
        else:
            between,within = stat_element

        result_df.append(stat_test(data_df,status_df,between,within,sample,**kwargs))
    return pd.concat(result_df,axis=1)
