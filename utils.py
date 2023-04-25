import numpy as np
import torch
from collections import Counter
from icecream import ic


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#---------- GPU information  ----------#
if torch.cuda.is_available():
    print(f"-- Current Device: {torch.cuda.get_device_name(0)}")
    print(
        f"-- Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("-- Let's use", torch.cuda.device_count(), "GPUs!")
    print(f"Device Name: {DEVICE}")
else:
    print("-- Unfortunately, we are only using CPUs now.")


def get_average_payoffs(payoff_lst, verbose=False):
    payoff_lst = np.array(payoff_lst)
    r0, r1 = np.mean(payoff_lst, axis=0)
    if verbose:
        print(line(60))
        print("Average Rewards")
        print(line(60))
        print(f"Agent 0 Average Reward: {r0}")
        print(f"Agent 1 Average Reward: {r1}")

    return r0, r1


def parse_payoffs(payoff_lst, verbose=False):
    # Assume that there are only two players
    # Can be extended to more players cases easily later
    payoff_lst = np.array(payoff_lst)
    p0_stats, p1_stats = payoff_lst[:,
                                    :1].squeeze(), payoff_lst[:, 1:].squeeze()
    assert p0_stats.shape == p1_stats.shape
    # ic(p0_stats.shape)
    p0_stats, p1_stats = Counter(list(p0_stats)), Counter(list(p1_stats))

    p0_wins, p0_lose, p0_draw = [p0_stats[r] for r in [1, -1, 0]]
    p1_wins, p1_lose, p1_draw = [p1_stats[r] for r in [1, -1, 0]]
    # Assertion check (not necessary but useful)
    n = len(payoff_lst)
    assert p0_draw == p1_draw
    assert p0_lose == p1_wins
    assert p1_lose == p0_wins
    assert p0_wins + p1_wins + p0_draw == n

    if verbose:
        print(line(60))
        print(f"Total Number of Games: {n}")
        print(
            f"Agent 0 wins {p0_wins} games (P0 win rate: {np.round(100 * p0_wins/n, 3)}%)")
        print(
            f"Agent 1 wins {p1_wins} games (P1 win rate: {np.round(100 * p1_wins/n, 3)}%)")
        print(
            f"Draws {p0_draw} games (Draw rate: {np.round(100 * p0_draw/n, 3)}%)")
    return p1_wins, p0_wins, p1_draw


def line(n):
    return "-"*n
