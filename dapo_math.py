"""DAPO objective components, kept separate for small CPU unit tests."""


def soft_overlong_penalty(length, max_length, buffer):
    if not 0 < buffer < max_length:
        raise ValueError("overlong_buffer must be between zero and max_completion_tokens")
    return -min(1.0, max(0.0, (length - (max_length - buffer)) / buffer))


def mixed_correctness(correct):
    return any(correct) and not all(correct)


def group_advantages(rewards):
    import torch
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)


def loss_sum(new_logps, old_logps, advantage, clip_low, clip_high):
    import torch
    ratio = torch.exp(new_logps - old_logps)
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    return -torch.minimum(ratio * advantage, clipped * advantage).sum()
