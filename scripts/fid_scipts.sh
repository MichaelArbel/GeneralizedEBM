# sbatch slurm_fid.sh configs/fids/sngan_imagenet_michael.yaml imh 
# sbatch slurm_fid.sh configs/fids/sngan_cifar10_michael.yaml imh 
# sbatch slurm_fid.sh configs/fids/sngan_lsun_michael.yaml imh 
# sbatch slurm_fid.sh configs/fids/sngan_celebA_michael.yaml imh 


#sbatch slurm_fid.sh configs/fids/sngan_imagenet_michael.yaml mala 
#sbatch slurm_fid.sh configs/fids/sngan_cifar10_michael.yaml mala 
#sbatch slurm_fid.sh configs/fids/sngan_lsun_michael.yaml mala 
#sbatch slurm_fid.sh configs/fids/sngan_celebA_michael.yaml mala 



#sbatch slurm_fid.sh configs/fids/sngan_imagenet_michael.yaml dot 
#sbatch slurm_fid.sh configs/fids/sngan_cifar10_michael.yaml dot 
#sbatch slurm_fid.sh configs/fids/sngan_lsun_michael.yaml dot
#sbatch slurm_fid.sh configs/fids/sngan_celebA_michael.yaml dot 

# sbatch slurm_fid.sh configs/fids/sngan_imagenet_michael.yaml mh 
# sbatch slurm_fid.sh configs/fids/sngan_cifar10_michael.yaml mh 
# sbatch slurm_fid.sh configs/fids/sngan_lsun_michael.yaml mh 
# sbatch slurm_fid.sh configs/fids/sngan_celebA_michael.yaml mh 

# array=( 10. 100. 1000. 0.01 0.1 1. ) 

# for temperature in ${array[@]}
# do
# 	echo " temperature $temperature "
# 	#sbatch scripts/slurm_fid.sh configs/sample/sngan_imagenet_michael.yaml mala $temperature
# 	#sbatch scripts/slurm_fid.sh configs/sample/sngan_cifar10_michael.yaml mala $temperature
# 	sbatch scripts/slurm_fid.sh configs/sample/sngan_lsun_michael.yaml mala $temperature
# 	#sbatch scripts/slurm_fid.sh configs/sample/sngan_celebA_michael.yaml mala $temperature

# done 

# array=( 0.01 0.1 1. ) 
# for temperature in ${array[@]}
# do
# 	echo " temperature $temperature "
# 	sbatch scripts/slurm_fid.sh configs/sample/sngan_celebA_michael.yaml mala $temperature

# done 


# array=( 1. 10. ) 
# for temperature in ${array[@]}
# do
# 	echo " temperature $temperature "
# 	sbatch scripts/slurm_fid.sh configs/sample/sngan_imagenet_michael.yaml mala $temperature

# done 


array=( 0.1 1. ) 

for temperature in ${array[@]}
do
	echo " temperature $temperature "
	#sbatch scripts/slurm_fid.sh configs/sample/sngan_imagenet_michael.yaml mala $temperature
	#sbatch scripts/slurm_fid.sh configs/sample/sngan_cifar10_michael.yaml mala $temperature
	sbatch scripts/slurm_fid.sh configs/sample/sngan_lsun_michael.yaml mala $temperature
	#sbatch scripts/slurm_fid.sh configs/sample/sngan_celebA_michael.yaml mala $temperature

done 












