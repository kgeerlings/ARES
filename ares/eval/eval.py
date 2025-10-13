def eval_agent_n_episodes(env : gym.Env, agent : AgentRL, n : int = 5, do_animation : bool = True) -> dict:
    return eval_policy_n_episodes(env, agent.act, n=n, do_animation=do_animation)