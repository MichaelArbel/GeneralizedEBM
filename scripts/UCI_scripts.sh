# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml redwine maf maf
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml whitewine maf maf
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml parkinsons maf maf







# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml redwine mogmaf mogmaf
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml whitewine mogmaf mogmaf
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml parkinsons mogmaf mogmaf

# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml redwine mogmaf mogmaf
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml whitewine mogmaf mogmaf
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml parkinsons mogmaf mogmaf


# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml redwine made made
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml whitewine made made
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml parkinsons made made

# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml redwine made made
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml whitewine made made
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml parkinsons made made












# sbatch scripts/UCI_slurm.sh configs/uci/redwine_donskin.yaml redwine maf maf 
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_donskin.yaml whitewine maf maf
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_donskin.yaml parkinsons maf maf

# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml redwine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml whitewine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml parkinsons nvp nvp


# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml redwine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml whitewine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_nll.yaml parkinsons nvp nvp

# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml redwine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml whitewine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml parkinsons nvp nvp

# sbatch scripts/UCI_slurm.sh configs/uci/redwine_donskin.yaml redwine nvp nvp 
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_donskin.yaml whitewine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_donskin.yaml parkinsons nvp nvp



# sbatch scripts/UCI_slurm.sh configs/uci/minibone.yaml minibone nvp nvp donsker 
# sbatch scripts/UCI_slurm.sh configs/uci/minibone.yaml hepmass nvp nvp ml
# sbatch scripts/UCI_slurm.sh configs/uci/minibone.yaml hepmass nvp nvp donsker



# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml redwine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml whitewine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_cd.yaml parkinsons nvp nvp

# sbatch scripts/UCI_slurm.sh configs/uci/minibone_nll.yaml minibone nvp nvp cd
# sbatch scripts/UCI_slurm.sh configs/uci/minibone_nll.yaml hepmass nvp nvp cd


# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml redwine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml whitewine nvp nvp
# sbatch scripts/UCI_slurm.sh configs/uci/redwine_kale.yaml parkinsons nvp nvp

# sbatch scripts/UCI_slurm.sh configs/uci/minibone_nll.yaml minibone nvp nvp kale
# sbatch scripts/UCI_slurm.sh configs/uci/minibone_nll.yaml hepmass nvp nvp kale



python main.py  --config=configs/uci/minibone_kale.yaml --dataset=minibone --generator=nvp --discriminator=nvp --criterion=donsker --device=0 --slurm_id=-1 &
python main.py  --config=configs/uci/minibone_kale.yaml --dataset=hepmass --generator=nvp --discriminator=nvp --criterion=donsker --device=1 --slurm_id=-1 &
# python main.py  --config=configs/uci/minibone_nll.yaml --dataset=hepmass --generator=nvp --discriminator=nvp --criterion=ml --device=2 --slurm_id=-1 &



#python main.py  --config=configs/uci/minibone_nll.yaml --dataset=minibone --generator=nvp --discriminator=nvp --criterion=cd --device=0  --slurm_id=-1 &
#python main.py  --config=configs/uci/minibone_nll.yaml --dataset=hepmass --generator=nvp --discriminator=nvp --criterion=cd --device=1 --slurm_id=-1 &

# python main.py  --config=configs/uci/redwine_cd.yaml --dataset=redwine --generator=nvp --discriminator=nvp --criterion=cd --device=1  --slurm_id=-1 &
# python main.py  --config=configs/uci/redwine_cd.yaml --dataset=whitewine --generator=nvp --discriminator=nvp --criterion=cd --device=2  --slurm_id=-1 & 
# python main.py  --config=configs/uci/redwine_cd.yaml --dataset=parkinsons --generator=nvp --discriminator=nvp --criterion=cd --device=0  --slurm_id=-1 & 



