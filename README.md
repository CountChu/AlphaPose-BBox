The project is forked from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).

The purpose of the project is to evaluate AlphaPose to generate skeletons. AlphaPose is a top-down human pose estimation. It uses yolo to detect ROIs in an image and generates skeletons from the ROIs. The problem is some ROIs with people can not be detected so that people's skeletons cannot be generated.

We suppose the perfect ROIs is required to generating the perfect skeletons. To evaluate the pose estimation of AlphaPose, we input an image with manual ROIs to test if AlphaPose generates correct skeletons in the ROIs.
