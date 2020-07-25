

How to copy the files to EC2 Instance?

sudo scp -r -i "hw3.pem" /Users/pedzindm/Desktop/school/sum20-Parallel-algs/assignments/hw3  ubuntu@ec2-18-234-171-144.compute-1.amazonaws.com:~/


How to connect to EC2 instance?

1)Navigate to assignment directory locally then

2) ssh -i "hw3.pem" ubuntu@ec2-18-234-171-144.compute-1.amazonaws.com


How to run cu file?

1) cd hw3/

2)nvcc -o hw3.out hw3.cu

2)./hw3.out 