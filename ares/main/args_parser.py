import argparse

def parse_args():
    """
    Parse command line arguments for training, fine-tuning, or evaluating the RL agent.
    """
    parser = argparse.ArgumentParser(description="ARES - Reinforcement Learning Training and Evaluation")

    # Required argument: type of operation
    parser.add_argument(
        "--type",
        required=True,
        choices=["train", "finetune", "eval"],
        help="Type of operation to perform: train, finetune, or eval.",
    )

    # Configuration choice
    parser.add_argument(
        "--config",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Configuration model to use (0, 1, 2, or 3). Corresponds to config, config_model_1, config_model_2, or config_model_3.",
    )

    # Evaluation specific arguments
    parser.add_argument("--checkpoint-path", type=str, help="Path to checkpoint file (for eval/finetune).")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes for evaluation.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum steps per episode.")
    parser.add_argument("--record", action="store_true", help="Record videos during evaluation.")
    parser.add_argument("--save-video", action="store_true", help="Save videos during evaluation.")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation.")

    return parser.parse_args()