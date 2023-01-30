# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:56:04 2022

A module for extending functionality of CNVPytor.

The module performs several things:
    1 - Rips out CNV calls from a list of given files and saves them to a file
    2 - Rips relevant read-depth data from a list of given files and saves it to one big file.
    3 - Genotypes given regions
    4 - Performs statistical testing based on a given column

Function 1 is performed by "write_cnv_call_file" function while functions
2,3,4 are performed sequentially by "statistical_pipeline" function.

===============================================================================
    WRITE CNV CALL FILE function
===============================================================================
The function takes following parameters:

    pytor_files : iterable of strings
        Contains path to the CNVPytor-made .hdf files.
    output_file : str
        Where to write the file. If 'None' is passed then
        the dataframe will simply be returned and not saved.
    ret : Bool,
        Return a dataframe.
    assembly_report : str, optional
        Path to assembly report. Assembly report is used to
        constrain chromosomes so that only RD data of
        assembled chromosomes is stored.

    binsize_dic : dict {str:int}
       If passed, retrieves binsize from the dictionary.
    flag_dic : dict {str:str}
        If passed, retrieves flagsize from the dictionary.

    lower_ratio, upper_ratio:
        Float limits of the ratio of mean Read Depth and its standard deviation.
        Used to find the optimal binsize. If "binsize_dic" is not
        passed, the ratio of RD mean and RD standard deviation is used
        to find optimal binsize, and therefore valid CNV calls.

===============================================================================
    STATISTICAL PIPELINE function
===============================================================================

Statistical pipeline function consists of 3 parts:
    1. Ripping read-depth data from a list of files and saving it to one file
    2. Genotyping data from a given dataframe that has genomic loci
    3. Performing Welch T-Test for the resulting values.


Statistical pipeline function takes the following non-optional argument:
    pytor_files : iterable of strings
        Contains path to the CNVPytor-made .hdf files.
    status : pd.DataFrame
        Categorical dataframe that will be used to perform statistical testing.
    assembly_report : str
        Path to assembly report. This is used to fetch data for only whole-chromosomes.
        'None' can be passed and chromosomes can be filtered via pandas operations later on.
    cnv_data : str or pd.DataFrame
        Dataframe of genomic regions that will be genotyped. Must contain columns
        "chrom","start","end". Alternatively, all CNV calls can be genotyped by passing
        either "all" in which case all CNVs found in samples are genotyped or
        "merged" in which overlapping CNVs are merged a la bedtools merge.

    output_rd : str
        Output for read-depth file. Must be specified.
    output_genotype : str
        Output for genotyped values. 'None' can be passed, in which
        case genotype values will not be saved.
    output_testing : str
        Output for results of statistical testing. 'None' can be passed, in which
        case testing will not be saved, only returned.

    stat_elements : an iterable of tuples.
        This determines sample grouping for the purpose of statistical testing.

    force_readfile : Bool
        If 'output_rd' already exists, the 1. function will be skipped, unless
        you specify True.
    force_genotype : Bool
        If 'output_genotype' already exists, the 2. function will be skipped, unless
        you specify True.

    write_kws : dictionary of keyword arguments related to function under number 1
    genotype_kws : dictionary of keyword arguments related to function under number 2
    stat_kws : dictionary of keyword arguments related to function under number 3


### 1. Ripping read-depth data ###
Optional keywords that are passed to a function for ripping read depth data
are:
    compress : str, optional
        Store in what kind of float datatype.
        "norm" for np.float32, "half" for np.float16, "double" for np.float64
        The default is "half".
    binsize_dic : dict {str:int}
       if passed, retrieves binsize from the dictionary
    flag_dic : dict {str:str}
        if passed, retrieves flagsize from the dictionary
    lower_ratio, upper_ratio:
        Float limits of the ratio of mean Read Depth and its standard deviation.
        Used to find the optimal binsize.

CNVPytor divides a genome into a 100 divisible bin sizes. One file can contain
data for more than one bin size, e.g. one can have data for bins of 100, 500, 800.
To determine which bin size will be fetched for any file, a dictionary can be passed.
For example, if you have a file "mortimer.pytor_output" in a folder named "cnvpytors"
that has bin sizes of 100, 500, 800, you can pass either:
    >>> sample_binsize = {"cnvpytors/mortimer.pytor_output":500}
    OR
    >>> sample_binsize = {"mortimer":500}

In the case that nothing is passed, then the algorithm will define a best binsize.
According to supplemental of CNVnator, the best binsize is selected when the ratio
of global read-depth mean to global read-depth standard deviation is between 4 or 5.
Alternatively, you can specifyy the ratio via 'lower_ratio' or 'upper_ratio'.

In the case that no binsize satisfies the above constraint, the one whose ratio
of read-depth mean to read-depth standard deviation is closest to the average
of 'lower_ratio' and 'upper_ratio' is selected.

One big caveat is that samples are saved without their extension or folder,
e.g. "cnvpytors/mortimer.pytor_output" becomes "mortimer".

For details, see 'reading.py' script.

### 2. Genotyping provided data ###
Optional keywords that are passed to a function that performs genotyping are:
    max_cores : int,
        Number of cores for parallel processing. The default is 1.

    samples : list of samples,
        All keys of the sample. The default is None (iterates through all samples)

For details on how genotyping works, see 'genotyping.py' script.

### 3. Statistical testing ###
To perform statistical testing, a categorical dataframe ('status'), an iterable
of tuples ('stat_elements'), and optional keywords are needed.

A categorical dataframe should contain one column (usually index, but the sample
column can be specified via 'sample' keyword) that contains sample names.
For example, if you have a CNVPytor made HDF file in "cnvpytors/mortimer.pytor_output",
this dataset is saved as "mortimer" inside the bulk read-depth file, and you should
have "mortimer" in the categorical dataframe.

An example of proper categorical dataframe:

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

To test samples between different dwellings, use the following code:
    >>> stat_elements = [("dwelling",None)]

To test samples between all "cave" regions and "surface" regions.
    >>> stat_elements = [("dwelling","region")]
To do both, use:
    >>> stat_elements = [("dwelling",None),("dwelling","region")]

P-values are adjusted automatically.
Optional keyword arguments that are passed to statistical testing are
    sample: str,
        Column containing samples in categorical dataframe.
        If samples are in indices of categorical dataframe, pass 'None'.
    row_method: str,
        Method used to adjust across rows
    col_method": str,
        Method used to adjust across columns.
    adjust_first : str,
        What to adjust first
    alpha: float,
        Level of statistical significance. The default is 0.05.
    effsize: str,
        What is used as effect size. Cohen's D ("D") or logfold change of means ("FC")

For details see 'stat_ttest.py' script


===============================================================================
    EXAMPLE CODE
===============================================================================
In the below block we import everything we need for processing files.
Glob is used to fetch out files, and we use logging to log everything we
have done in file specified in 'filename'.

    >>> import glob
    >>> import logging
    >>> logging.basicConfig(level=logging.INFO,
    >>>                     filename="result_folder/log.txt")
    >>> import pandas as pd
    >>> from genotyper import statistical_pipeline

Below we define inputs of the function.
We use glob to fetch all CNVPytor made HDF files we stored in one folder,
and we likewise read a categorical dataframe that has information about
all our samples. Information is found in "Sample Name" column, so we
quickly set it as index.

We have information about assembly report in a text file, so we
pass it in. We want to genotype all merged CNV regions so we specify "merged" as
'cnv_data'. We want to test all "surface" regions against all "cave" regions
and all pooled "surface" sampels against all pooled "cave" samples so we specify that
in 'stat_elements'.

    >>> pytor_files = glob.glob("pytor/*.pytor")
    >>> status = pd.read_csv("linker.csv")[["Sample Name","dwelling","lineage","region"]]
    >>> status = status.set_index("Sample Name")
    >>> assembly_report = "assembly_report.txt"
    >>> cnv_data = "merged"
    >>> stat_elements = [("dwelling","region"),("dwelling",None)]

We want to save our read depth and statistical testing files.
We don't need output of genotypes since it's already found in
testing.

    >>> output_rd = "result_folder/read_depth_file.h5"
    >>> output_genotype = None
    >>> output_testing = "result_folder/stat_test.csv"

We don't need to modify writing, but we'd like to speed up genotypization
so we maximize the number of cores we use. For statistical testing we'll
set alpha to 0.01 and use fold change as effect size measure.

    >>> write_kws = None
    >>> genotype_kws = {"max_cores":8}
    >>> stat_kws = {"alpha":0.01,"effsize":"FC"}

Lastly, we'll call our function.
Our function is called within 'if __name__=="__main__"' block to
prevent problems resulting from multithreading module.

    >>> if __name__=="__main__":
    >>>     data = statistical_pipeline(pytor_files = pytor_files,
    >>>                                 status = status,
    >>>                                 assembly_report = assembly_report,
    >>>                                 cnv_data = cnv_data,
    >>>                                 stat_elements = stat_elements,
    >>>                                 output_rd=output_rd,
    >>>                                 output_genotype=output_genotype,
    >>>                                 output_testing=output_testing,
    >>>                                 write_kws = write_kws,
    >>>                                 genotype_kws = genotype_kws,
    >>>                                 stat_kws = stat_kws
    >>>                                 )

The output log can be checked, and we see that the entire procedure took roughly
420 seconds for 15k independent regions and 42 samples.

@author: ivanp
"""
import time
import os
import logging
import pandas as pd
from .reading import write_read_depth_file,write_cnv_call_file
from .genotyping import genotype
from .stat_ttest import stat_test_wrangler
logging.basicConfig(level=logging.INFO)


def merge_on_locus(data,chromosomes=None,cols=None):
    """
    Merges overlapping intervals to get merged data

    Adapted from https://stackoverflow.com/questions/57882621

    Parameters
    ----------
    data : pd.DataFrame.
    chromosomes : iterable, optional
        List of chromosomes on which to merge.
        The default is None meaning it merges on all unique
        chromosomes of found in "chrom" column
    cols : iterable, optional
        The locus parts of dataframe,
        where chromosomes are, and starts and ends are
        The default is None meaning ["chrom","start","end"]

    Returns
    -------
    merged dataframe

    """
    if cols is None:
        cols = ["chrom","start","end"]
    _chrom_col,_start_col,_end_col = cols
    if chromosomes is None:
        chromosomes = data[_chrom_col].unique().tolist()

    #sort the data - without this, nothing works
    data = data.sort_values([_chrom_col,_start_col])

    result_df = list()

    _s = time.perf_counter()
    for chrom in chromosomes:
        input_df = data[data[_chrom_col]==chrom].copy()
        input_df[_start_col]=input_df[_start_col].astype(float).astype(int)
        input_df[_end_col]=input_df[_end_col].astype(float).astype(int)

        input_df["group"]=(input_df[_start_col]>input_df[_end_col].shift().cummax()).cumsum()
        merged = input_df.groupby("group").agg({_start_col:"min",_end_col:"max"})
        merged.insert(0,_chrom_col,chrom)
        logging.info("Merged data on chromosome %s, went from %s to %s",chrom,len(input_df),len(merged))
        result_df.append(merged)
    _e = time.perf_counter()
    logging.info("Merge done in %.4f seconds",(_e-_s))
    return pd.concat(result_df,axis=0,ignore_index=True)

def statistical_pipeline(pytor_files,status,assembly_report,cnv_data,
                         stat_elements,output_rd,output_genotype,output_testing,
             force_readfile=False,force_genotype=False,
             write_kws = None,genotype_kws = None, stat_kws = None):
    """
    pytor_files : iterable of strings
        Contains path to the CNVPytor-made .hdf files.
    status : pd.DataFrame
        Categorical dataframe that will be used to perform statistical testing.
    assembly_report : str
        Path to assembly report. This is used to fetch data for only whole-chromosomes.
        'None' can be passed and chromosomes can be filtered via pandas operations later on.
    cnv_data : str or pd.DataFrame
        Dataframe of genomic regions that will be genotyped. Must contain columns
        "chrom","start","end". Alternatively, all CNV calls can be genotyped by passing
        either "all" in which case all CNVs found in samples are genotyped or
        "merged" in which overlapping CNVs are merged a la bedtools merge.

    stat_elements : an iterable of tuples.
        This determines sample grouping for the purpose of statistical testing.

    output_rd : str
        Output for read-depth file. Must be specified.
    output_genotype : str
        Output for genotyped values. 'None' can be passed, in which
        case genotype values will not be saved.
    output_testing : str
        Output for results of statistical testing. 'None' can be passed, in which
        case testing will not be saved, only returned.



    force_readfile : Bool
        If 'output_rd' already exists, the 1. function will be skipped, unless
        you specify True.
    force_genotype : Bool
        If 'output_genotype' already exists, the 2. function will be skipped, unless
        you specify True.

    write_kws : dictionary of keyword arguments related to
        ripping read depth to the specified read-depth file
    genotype_kws : dictionary of keyword arguments related to
        genotyping regions outlined in the  given "cnv_data"
    stat_kws : dictionary of keyword arguments related to
        statistical testing
    """
    _s = time.perf_counter()
    #check for provided pytor files and cnv data
    if pytor_files is None and isinstance(cnv_data,str):
        raise ValueError("If pytor files are not provided, CNV data cannot be inferred from a string")
    #parse dictionaries
    write_kws = dict() if write_kws is None else write_kws
    genotype_kws = dict() if genotype_kws is None else genotype_kws
    stat_kws = dict() if stat_kws is None else stat_kws


    #parts related to writing a huge RD file
    if not os.path.isfile(output_rd) or force_readfile:
        write_read_depth_file(pytor_files,output_rd,assembly_report=assembly_report,**write_kws)
        logging.info("Ripping RD from provided HDF files...")
    else:
        logging.info("Found RD HDF file, skipping writing...")

    #parts related to genotyping
    if isinstance(cnv_data,pd.DataFrame):
        logging.info("Using pandas dataframe of size %s for genotype",len(cnv_data))
        if not all(item in cnv_data.columns for item in ["chrom","start","end"]):
            raise ValueError("CNV data must contain columns 'chrom', 'start', 'end'")


    elif cnv_data == "all":
        logging.info("Using all CNV calls as CNV data for genotype...")
        cnv_data = write_cnv_call_file(pytor_files,output=None,ret=True,assembly_report=assembly_report)

    elif cnv_data == "merged":
        logging.info("Using merged CNV calls as CNV data for genotype...")
        cnv_data = write_cnv_call_file(pytor_files,output=None,ret=True,assembly_report=assembly_report)
        cnv_data = merge_on_locus(cnv_data)
    else:
        raise ValueError("Invalid value for parameter 'cnv_data'")

    if output_genotype:
        if not os.path.isfile(output_genotype) or force_genotype:
            logging.info("Either forcing or genotype not found, genotyping...")
            genotyped = genotype(output_rd,cnv_data,output_genotype,**genotype_kws)
        else:
            logging.info("Found genotype data, skipping genotyping...")
            genotyped = pd.read_csv(output_genotype)
    else:
        genotyped = genotype(output_rd,cnv_data,output_genotype,**genotype_kws)


    #parts related to statistical testing
    logging.info("Performing statistical testing...")
    result = stat_test_wrangler(genotyped,status,*stat_elements,**stat_kws)
    _e = time.perf_counter()
    logging.info("TOTAL TIME ELAPSED : %.4f seconds",(_e-_s))
    if output_testing:
        result.to_csv(output_testing)
        logging.info("Saved the results at %s",output_testing)
    return result
