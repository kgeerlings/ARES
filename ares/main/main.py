from ares.main.args_parser import parse_args
from ares.main.global_variables import GlobalVariables


def main():
    """
    Main function to execute training, fine-tuning, or evaluation based on arguments.
    """
    args = parse_args()

    # Select configuration based on --config argument
    from config.config import config, config_model_1, config_model_2, config_model_3
    
    config_map = {
        0: config,
        1: config_model_1,
        2: config_model_2,
        3: config_model_3,
    }
    selected_config = config_map[args.config]
    print(f"Using configuration: config_model_{args.config}")
    GlobalVariables.CONFIG = args.config


    if args.type == "train":
        print("Starting training...")
        from ares.training.train import Trainer
        trainer = Trainer()
        trainer.train()

    elif args.type == "finetune":
        if not args.checkpoint_path:
            print("Error: --checkpoint-path is required for fine-tuning.")
            return
        print(f"Starting fine-tuning from checkpoint: {args.checkpoint_path}")
        from ares.training.finetune_model import FineTuner
        finetuner = FineTuner()
        finetuner.train()

    elif args.type == "eval":
        if not args.checkpoint_path:
            print("Error: --checkpoint-path is required for evaluation.")
            return
        print(f"Starting evaluation from checkpoint: {args.checkpoint_path}")
        from ares.eval.eval import Evaluator
        evaluator = Evaluator(env_choice=args.config, configuration=selected_config)
        evaluator.run(
            checkpoint_path=args.checkpoint_path,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            record=args.record,
            save_video=args.save_video
        )


if __name__ == "__main__":
    main()
