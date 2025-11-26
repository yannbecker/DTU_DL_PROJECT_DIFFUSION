import subprocess

################################################# Default parameters ################################################# 

defaults = dict(
        data_dir="/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad",
        val_data_dir="",
        noised=True,
        iterations=500000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=128,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=100,
        eval_interval=100,
        save_interval=100000,
        vae_path='output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt',
        latent_dim=128,
        model_path='output/classifier_checkpoint/classifier_muris',
        start_guide_time=500,
        num_class=5, ########### /!\ à adapter à la condition key (leiden -> 5 classes)
    )

############################################ Chosen parameters ########################################################

# Paths 
model_path = defaults["model_path"] # MODIFY
vae_path = "/zhome/f0/d/223076/Projet_Deep_Learning/DTU_DL_PROJECT_DIFFUSION/src/VAE/output/ae_checkpoint/vae_bulk_transcript_pca/model_seed=0_step=1999.pt"
data_dir = defaults["data_dir"] # MODIFY
val_data_dir = defaults["val_data_dir"]

# Training parameters 
iterations = defaults["iterations"]

lr = defaults["lr"]
anneal_lr = defaults["anneal_lr"]

latent_dim = defaults["latent_dim"]
batch_size = defaults["batch_size"]

weight_decay = defaults["weight_decay"]

# Others             
noised = defaults["noised"]
microbatch = defaults["microbatch"]
schedule_sampler = defaults["schedule_sampler"]
resume_checkpoint = defaults["resume_checkpoint"]
log_interval = defaults["log_interval"]
eval_interval = defaults["eval_interval"]
save_interval = defaults["save_interval"]
start_guide_time = defaults["start_guide_time"]
num_class = defaults["num_class"]
classifier_use_fp16 = defaults["classifier_use_fp16"]

################################# Parameters for classifier_train.py #####################################

args = [
    "--data_dir", data_dir,
    "--val_data_dir", val_data_dir,
    "--noised", noised,
    "--iterations", iterations,
    "--lr", lr,
    "--weight_decay", weight_decay,
    "--anneal_lr", anneal_lr,
    "--batch_size", batch_size,
    "--microbatch", microbatch,
    "--schedule_sampler", schedule_sampler,
    "--resume_checkpoint", resume_checkpoint,
    "--log_interval", log_interval,
    "--eval_interval", eval_interval,
    "--save_interval", save_interval,
    "--vae_path", vae_path,
    "--latent_dim", latent_dim,
    "--model_path", model_path,
    "--start_guide_time", start_guide_time,
    "--num_class", num_class,
    "--classifier_use_fp16", classifier_use_fp16,
]


######################################## Run classifier_train.py ###################################

if __name__ == "__main__":
    subprocess.run(["python", "src.classifier.classifier_train.py"] + args, check=True)




