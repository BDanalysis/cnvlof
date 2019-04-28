# CNV-LOF
CNV-LOF is designed to detect copy number variation using sequencing
data from human

## Installation
The following software must be installed on your machine:
Python : tested with version 3.6
R: tested with version 3.5

### Python dependencies
* numpy 
* numba
* sklearn
* pandas
* pysam
* rpy2

You can install the above package using the following command：
pip install numpy numba sklearn pandas pysam rpy2


### R packages
* DNAcopy

To install this package, start R and enter:
>install.packages("BiocManager")
>BiocManager::install("DNAcopy")


## Running
CNV-LOF requires two input files, a bam file and a reference folder,
the folder contains the reference sequence for each chromosome of the bam file。

### runnig command
python CNV-LOF.py [bamfile] [reference] [output] [binSize] [segCount] [k]

·bamfile: a bam file
·reference: the reference folder path
·output:the the output file path
·binSize: the window size ('1000'by default)
·segCount: the number of  partitions to CBS ('50' by default)·
·K:the parameters of the LOF ('10' by default)

### run the default example
python CNV-LOF.py test.bam /Users/jong/oc_svm/reference /Users/jong/oc_svm/result 1000 50 10 

Finally, two result files will be generated, one is the mutation information of detection and the other is the statistical data





