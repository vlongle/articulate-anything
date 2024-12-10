from typing import Callable, Any, Dict, Optional
from omegaconf import DictConfig
import logging
import traceback
from articulate_anything.utils.utils import (
    join_path,
    string_to_file,
    create_dir,
    Steps,
)


def error_handler(e: Exception, task, iteration: int, seed: int, cfg: DictConfig):
    """Handle errors in the actor-critic loop."""
    link_err_dir = join_path(cfg.out_dir, task,
                             f"iter_{iteration}", f"seed_{seed}")
    create_dir(link_err_dir)
    string_to_file(traceback.format_exc(),
                   join_path(link_err_dir, "error.txt"))


def default_pick_best(results):
    """
    Pick the latest iteration with the highest score and the lowest seed number
    """
    sorted_results = sorted(
        results,
        key=lambda x: (-x["iteration"], -x["feedback_score"], x["seed"]),
    )
    return sorted_results[0] if sorted_results else None


def default_load_result_func(best_result):
    return best_result


def actor_critic_loop(
    cfg: DictConfig,
    actor_func: Callable[[int, int, Dict[str, Any]], Any],
    critic_func: Callable[[Any], int],
    steps: Steps,
    pick_best_func: Optional[Callable[[list], Any]] = None,
    load_result_func: Optional[Callable[[Any], Dict[str, Any]]] = None,
    error_handler: Callable[[Exception, int, int], None] = None,
    post_process_iter: Optional[Callable[[Any], Any]] = None,
    retry_kwargs={}
) -> Any:
    results = []

    # Use default pick_best function if not provided
    if pick_best_func is None:
        pick_best_func = default_pick_best
    if load_result_func is None:
        load_result_func = default_load_result_func

    for iteration in range(cfg.actor_critic.max_iter):
        for seed in range(cfg.actor_critic.num_seeds):
            try:
                # Run actor
                actor_result = actor_func(iteration, seed, retry_kwargs)
                # Run critic
                critic_result = critic_func(iteration, seed, actor_result)
                feedback_score = critic_result["feedback_score"]
            except Exception as e:
                logging.error(
                    f"Failed to run actor-critic loop for iteration {iteration}, seed {seed}:")
                logging.error(traceback.format_exc())
                if error_handler:
                    error_handler(e, iteration, seed)
                feedback_score = -1
                actor_result = {}
                critic_result = {"feedback_score": -1}

            step_result = {"iteration": iteration, "seed": seed}
            step_result.update(actor_result)
            step_result.update(critic_result)

            results.append(step_result)

            best_result = pick_best_func(results)

            if post_process_iter:
                post_process_iter(best_result, cfg, steps)

            logging.info(
                f"At iteration {iteration}, seed {seed}, Best result so far {best_result}")

            if best_result['feedback_score'] > cfg.actor_critic.cutoff:
                logging.info(f">>> Success with result {best_result}")
                return best_result

            retry_kwargs = load_result_func(best_result)

            if cfg.actor_critic.conservative and feedback_score >= 0:
                logging.info(
                    f"Found successful seed {seed}, terminating iteration {iteration} early.")
                break  # Stop running more seeds for this iteration

    return pick_best_func(results)
