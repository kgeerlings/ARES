import argparse
from ares.training.train import Trainer
from ares.training.finetune_model import FineTuner
from ares.eval.eval import Evaluator

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

    # Hyperparameters (optional)
    parser.add_argument("--frames-per-batch", type=int, help="Number of frames per batch.")
    parser.add_argument("--minibatch-size", type=int, help="Size of minibatch.")
    parser.add_argument("--clip-epsilon", type=float, help="PPO clip epsilon.")
    parser.add_argument("--entropy-coef", type=float, help="Entropy coefficient.")
    parser.add_argument("--critic-coef", type=float, help="Critic coefficient.")
    parser.add_argument("--gamma", type=float, help="Discount factor gamma.")
    parser.add_argument("--lmbda", type=float, help="GAE lambda.")
    parser.add_argument("--max-grad-norm", type=float, help="Max gradient norm for clipping.")
    parser.add_argument("--learning-rate", type=float, help="Learning rate.")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs per iteration.")

    # Evaluation specific arguments
    parser.add_argument("--checkpoint-path", type=str, help="Path to checkpoint file (for eval/finetune).")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes for evaluation.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum steps per episode.")
    parser.add_argument("--record", action="store_true", help="Record videos during evaluation.")
    parser.add_argument("--save-video", action="store_true", help="Save videos during evaluation.")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation.")

    return parser.parse_args()


def main():
    """
    Main function to execute training, fine-tuning, or evaluation based on arguments.
    """
    args = parse_args()

    # Build hyperparameters dict from provided arguments
    hyperparams = {}
    if args.frames_per_batch is not None:
        hyperparams["frames_per_batch"] = args.frames_per_batch
    if args.minibatch_size is not None:
        hyperparams["minibatch_size"] = args.minibatch_size
    if args.clip_epsilon is not None:
        hyperparams["clip_epsilon"] = args.clip_epsilon
    if args.entropy_coef is not None:
        hyperparams["entropy_coef"] = args.entropy_coef
    if args.critic_coef is not None:
        hyperparams["critic_coef"] = args.critic_coef
    if args.gamma is not None:
        hyperparams["gamma"] = args.gamma
    if args.lmbda is not None:
        hyperparams["lmbda"] = args.lmbda
    if args.max_grad_norm is not None:
        hyperparams["max_grad_norm"] = args.max_grad_norm
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        hyperparams["num_epochs"] = args.num_epochs

    if args.type == "train":
        print("Starting training...")
        trainer = Trainer(**hyperparams)
        trainer.train()

    elif args.type == "finetune":
        if not args.checkpoint_path:
            print("Error: --checkpoint-path is required for fine-tuning.")
            return
        print(f"Starting fine-tuning from checkpoint: {args.checkpoint_path}")
        finetuner = FineTuner(**hyperparams)
        finetuner.train()

    elif args.type == "eval":
        if not args.checkpoint_path:
            print("Error: --checkpoint-path is required for evaluation.")
            return
        print(f"Starting evaluation from checkpoint: {args.checkpoint_path}")
        evaluator = Evaluator()
        evaluator.run(
            checkpoint_path=args.checkpoint_path,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            record=args.record,
            save_video=args.save_video
        )


if __name__ == "__main__":
    main()
