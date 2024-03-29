# MRtoUS

CNN to generate US from brain MR images.

## Installation

1. Download *Group 2* from http://www.bic.mni.mcgill.ca/~laurence/data/data.html
2. Download and install *MINC Toolset* from http://bic-mni.github.io

## Pre Processing

1. Create transformation `register -sync 01_mr_tal.mnc 01a_us_tal.mnc 01_all.tag`
2. Apply transformation `mincresample 01a_us_tal.mnc 01_us_reg.mnc -transformation 01_reg.xfm -like 01_mr_tal.mnc`
3. Convert to MINC2 (HDF5) `mincconvert -2 01_mr_tal.mnc 01_mr.mnc`
4. Execute `convert.ipynb` to create `0?.tfrecord` files

You can also download the pre-processed data:

* https://syncandshare.lrz.de/dl/fiFScpT6Gm1B3nDCotd52en7/mnibite-minc-processed.tar.bz2
* https://syncandshare.lrz.de/dl/fi3zNjQ49kfPhJEPabpMrZvN/mnibite-tfrecord-processed-threshold-10.tar.bz2
* https://syncandshare.lrz.de/dl/fiCTvUTd1aRJijtjMG5o1ryg/mnibite-tfrecord-processed-threshold-30.tar.bz2

The first contains the MR and US volumes in MINC2 format (applied steps 1. to 3.) the two other ones contain
7x7 patches of US and MR slices in tfrecord format but with different threshold value.

**threshold-10**

| ### | patches (th-10) | patches (th-20) |
| --- | ---------------:| ---------------:|
|  01 |         3093549 |          129635 |
|  02 |         2521154 |          134296 |
|  03 |         3425236 |          154154 |
|  04 |         1671087 |          153753 |
|  05 |         1620205 |           48974 |
|  06 |         1808115 |           19861 |
|  07 |         2151729 |           55009 |
|  08 |         3379828 |          514342 |
|  09 |         2043328 |          269513 |
|  10 |         2240942 |           85392 |
|  11 |         2647268 |          188275 |
|  12 |         1614372 |           36305 |
|  13 |         2681649 |           74709 |
|  14 |         3248508 |          163692 |
|     |    **34146970** |     **2027910** |

The threshold is defined in `convert.ipynb` and is used to filter out patches where there is no or weak US available.

## License

Copyright 2016 Bodo Kaiser <bodo.kaiser@me.com>
