## Setup for EC2 Instance:
1) Launch a P3.2XLarge with Deep Learning Ubuntu18.04 AMI
2) Follow this https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html
3) Follow this https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions

## How to copy the files to EC2 Instance?

sudo scp -r -i "hw3.pem" /Users/pedzindm/Desktop/school/sum20-Parallel-algs/assignments/hw3  ubuntu@ec2-3-237-19-2.compute-1.amazonaws.com:~/


## How to connect to EC2 instance?

1) Navigate to assignment directory locally

2) ssh -i "hw3.pem" ubuntu@ec2-3-237-19-2.compute-1.amazonaws.com



## How to run cuda file?

1) cd hw3/

2) nvcc -o `fileName`.out `filename`.cu

2) ./`fileName`.out 

WARNING: THe ubuntu@ec part can change so if you are having issues let me know

## How to create new inp.txt files?

1) navigate to Hw3
2) python3 inputGen/dataGenerator.py 1001050 999
Arg1 = number of Values; Arg2 = range of values... ^ would return 1001050 numbers with range of 0-999

## How to download all the data files?

aws s3 sync s3://dale-aviation-graphs .
