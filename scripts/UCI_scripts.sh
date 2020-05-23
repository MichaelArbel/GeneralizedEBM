# sbatch UCI_slurm.sh configs/redwine_cd.yaml redwine 
# sbatch UCI_slurm.sh configs/redwine_cd.yaml whitewine 
# sbatch UCI_slurm.sh configs/redwine_cd.yaml parkinsons 


# sbatch UCI_slurm.sh configs/redwine_nll.yaml redwine 
# sbatch UCI_slurm.sh configs/redwine_nll.yaml whitewine 
# sbatch UCI_slurm.sh configs/redwine_nll.yaml parkinsons 


sbatch UCI_slurm.sh configs/redwine_kale.yaml redwine 
sbatch UCI_slurm.sh configs/redwine_kale.yaml whitewine 
sbatch UCI_slurm.sh configs/redwine_kale.yaml parkinsons 


sbatch UCI_slurm.sh configs/redwine_donskin.yaml redwine 
sbatch UCI_slurm.sh configs/redwine_donskin.yaml whitewine 
sbatch UCI_slurm.sh configs/redwine_donskin.yaml parkinsons 



# sbatch UCI_slurm.sh configs/redwine_kale.yaml parkinsons 
# sbatch UCI_slurm.sh configs/redwine_donskin.yaml parkinsons 





# sbatch UCI_slurm.sh configs/redwine_kale.yaml redwine 
# sbatch UCI_slurm.sh configs/redwine_kale.yaml whitewine 
# sbatch UCI_slurm.sh configs/redwine_kale.yaml parkinsons 


# python run_energy.py --config=configs/redwine_nll.yaml --data_name=redwine --device=1 & 
# python run_energy.py --config=configs/redwine_nll.yaml --data_name=parkinsons --device=0 &
# python run_energy.py --config=configs/redwine_nll.yaml --data_name=whitewine --device=1 &

















