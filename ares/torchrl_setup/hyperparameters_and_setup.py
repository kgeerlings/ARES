from ares.main.global_variables import GlobalVariables

if GlobalVariables.CONFIG == 0:
    from config.config import config
elif GlobalVariables.CONFIG == 1:
    from config.config import config_model_1 as config
elif GlobalVariables.CONFIG == 2:
    from config.config import config_model_2 as config
elif GlobalVariables.CONFIG == 3:
    from config.config import config_model_3 as config

max_grad_norm = config["hyperparameters_and_setup"]["max_grad_norm"]
clip_epsilon = config["hyperparameters_and_setup"]["clip_epsilon"]
entropy_coef = config["hyperparameters_and_setup"]["entropy_coef"]
critic_coef = config["hyperparameters_and_setup"]["critic_coef"]
loss_critic_type = config["hyperparameters_and_setup"]["loss_critic_type"]
normalize_advantage = config["hyperparameters_and_setup"]["normalize_advantage"]
gamma = config["hyperparameters_and_setup"]["gamma"]
lmbda = config["hyperparameters_and_setup"]["lmbda"]
use_entropy_loss = config["hyperparameters_and_setup"]["use_entropy_loss"]
learning_rate = config["hyperparameters_and_setup"]["learning_rate"]
num_epochs = config["hyperparameters_and_setup"]["num_epochs"]
device = config["hyperparameters_and_setup"]["device"]
num_cells = config["hyperparameters_and_setup"]["num_cells"]
num_cells_critic = config["hyperparameters_and_setup"]["num_cells_critic"]
frames_per_batch = config["hyperparameters_and_setup"]["frames_per_batch"]
minibatch_size = config["hyperparameters_and_setup"]["minibatch_size"]
total_frames = config["hyperparameters_and_setup"].get("total_frames", 6000000)
