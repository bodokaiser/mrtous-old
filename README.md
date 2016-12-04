# MRtoUS

CNN to generate US from brain MR images.

## Installation

1. Download *Group 2* from http://www.bic.mni.mcgill.ca/~laurence/data/data.html
2. Download and install *MINC Toolset* from http://bic-mni.github.io

## Pre Processing

1. Create transformation `register -sync 01_mr_tal.mnc 01a_us_tal.mnc 01_all.tag`
2. Apply transformation `mincresample 01a_us_tal.mnc 01_us_reg.mnc -transformation 01_reg.xfm -like 01_mr_tal.mnc`
3. Convert to MINC2 (HDF5) `mincconvert -2 01_mr_tal.mnc 01_mr.mnc`

## License

Copyright 2016 Bodo Kaiser <bodo.kaiser@me.com>
