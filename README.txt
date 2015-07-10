This is implementation of Context-Conditioned Component Analysis described in:
@InProceedings{Turmukhambetov_2015_CVPR,
author = {Turmukhambetov, Daniyar and Campbell, Neill D.F. and Prince, Simon J.D. and Kautz, Jan},
title = {Modeling Object Appearance Using Context-Conditioned Component Analysis},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2015}
}
If you use this code, please cite the reference above.


The "CCCA.m" file contains the implementation of the model. Images and results of the model are generated using "get_visualization.m" function. Optionally, you can use "mex_omp_smm.cpp" file for faster "sparse matrix times full(dense) matrix multiplication" operation . Please use "-largeArrayDims" and "/openmp" (or "-fopenmp") flags when compiling.


The model is demonstrated on 4 datasets. Please run corresponding "demo_*.m" file to generate results.


The context vectors of the datasets are computed in "prep_dataset.m" file. Please edit the paths for datasets in the file accordingly.


For elephants, please download "UCL Parts Dataset" from: 
http://visual.cs.ucl.ac.uk/pubs/ccca/


For horses, please download "UCL Parts Dataset" from: 
http://visual.cs.ucl.ac.uk/pubs/ccca/
The images of horses from the dataset are from "Weizmann Horse Database". So, if you use the images or segmentations of horses, cite one of the papers listed in:
http://www.msri.org/people/members/eranb/


For cats, please download "PASCAL-Part Dataset" from:
http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html
and cite 
@InProceedings{chen_cvpr14,
 author       = {Xianjie Chen and Roozbeh Mottaghi and Xiaobai Liu and Sanja Fidler and Raquel Urtasun and Alan Yuille},
 title        = {Detect What You Can: Detecting and Representing Objects using Holistic Models and Body Parts},
 booktitle    = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year         = {2014},
}
Also, download original images from PASCAL VOC 2010 Dataset from: 
http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html
and cite
@misc{pascal-voc-2010,
 author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
 title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2010 {(VOC2010)} {R}esults",
 howpublished = "http://www.pascal-network.org/challenges/VOC/voc2010/workshop/index.html",
}

		
For facades, please download "CVPR 2010 Dataset" from:
http://vision.mas.ecp.fr/Personnel/teboul/data.php
and cite the webpage referred as "Ecole Centrale Paris Facades Database".


Feel free to contact Daniyar Turmukhambetov if you have any questions.