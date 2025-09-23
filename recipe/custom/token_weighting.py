# === Token Weighting Module ===
"""
Token weighting module for reinforcement learning training.
This module provides functionality to assign different weights to tokens
based on their semantic importance in the training process.
"""

import json
import re
from typing import Dict, List, Tuple, Union

import torch


# Constants for weight adjustment
DEFAULT_GAMMA_FORMAT = 1.0
DEFAULT_STEP_WEIGHT_INCREASE = 0.002
DEFAULT_BASE_THINK_WEIGHT = 0.6
DEFAULT_MAX_GAMMA_ARGS = 1.2
DEFAULT_NORMALIZATION_EPS = 1e-8

# Regex patterns for tag matching
THINK_TAG_PATTERN = r"<think>(.*?)</think>"
RESPONSE_TAG_PATTERN = r"<response>(.*?)</response>"
TOOL_CALL_TAG_PATTERN = r"<tool_call>(.*?)</tool_call>"
TOOL_CALL_CONTENT_PATTERN = r"(?:<tool_call>|<toolcall>)(.*?)(?:</tool_call>|</toolcall>)"
TOOL_NAME_PATTERN = r'"name"\s*:\s*"([^"]+)"'
TOOL_ARGS_PATTERN = r'"(?:arguments|parameters)"\s*:\s*'


def char_span_to_token_span(
    text: str, 
    char_lo: int, 
    char_hi: int, 
    tokenizer
) -> Tuple[int, int]:
    """Convert character span to token span using tokenizer.
    
    Args:
        text: Input text string
        char_lo: Character start position
        char_hi: Character end position  
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple of (token_start, token_end) positions
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    lo = next(i for i, (a, b) in enumerate(offsets) if a >= char_lo)
    hi = max(lo + 1, next((i for i, (a, b) in enumerate(offsets) if b >= char_hi), len(offsets)))
    return lo, hi

def _find_braced_span(s: str, open_idx: int) -> Tuple[int, int]:
    """Find the matching closing brace for a JSON object.
    
    Args:
        s: Input string
        open_idx: Index of the opening brace
        
    Returns:
        Tuple of (start_idx, end_idx) for the complete braced span
    """
    depth = 0
    i = open_idx
    in_str = False
    esc = False
    
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return open_idx, i + 1
        i += 1
    return open_idx, open_idx  # No matching brace found


def _assign_tag_weights(
    text: str,
    tag_pattern: str,
    token_weight: torch.Tensor,
    row_idx: int,
    weight_value: float,
    tokenizer
) -> None:
    """Assign weights to opening and closing tags.
    
    Args:
        text: Input text
        tag_pattern: Regex pattern for tag matching
        token_weight: Token weight tensor to update
        row_idx: Row index in the batch
        weight_value: Weight value to assign
        tokenizer: Tokenizer instance
    """
    for m in re.finditer(tag_pattern, text, flags=re.S):
        # Assign weight to opening tag
        start_tag_lo, start_tag_hi = m.span(0)
        start_tag_end = start_tag_lo + text[start_tag_lo:start_tag_hi].find('>') + 1
        t_lo, t_hi = char_span_to_token_span(text, start_tag_lo, start_tag_end, tokenizer)
        if t_hi > t_lo:
            token_weight[row_idx, t_lo:t_hi] = token_weight[row_idx, t_lo:t_hi].maximum(
                torch.full((t_hi - t_lo,), weight_value, device=token_weight.device)
            )
        
        # Assign weight to closing tag
        end_tag_start = start_tag_lo + text[start_tag_lo:start_tag_hi].rfind('<')
        end_tag_hi = start_tag_hi
        t_lo, t_hi = char_span_to_token_span(text, end_tag_start, end_tag_hi, tokenizer)
        if t_hi > t_lo:
            token_weight[row_idx, t_lo:t_hi] = token_weight[row_idx, t_lo:t_hi].maximum(
                torch.full((t_hi - t_lo,), weight_value, device=token_weight.device)
            )


def build_token_weight(
    batch: Dict[str, Union[torch.Tensor, List[str], int]], 
    tokenizer,
    gamma_name: float = 1.2, 
    gamma_args: float = 0.6,
    alpha: float = 0.5, 
    beta: float = 5.0,
    w_min: float = 0.001, 
    w_max: float = 8.0,
) -> torch.Tensor:
    """Build token weights for reinforcement learning training.
    
    Assigns weights to tokens in <tool_call>/<toolcall> blocks:
    - "name" field gets moderate weight
    - "arguments"/"parameters" field gets highest weight  
    - Format parts (think, response, tool_call tags) get moderate weight
    - Other tokens get zero weight
    
    Args:
        batch: Dictionary containing:
            - "loss_mask": Tensor of shape (B, T) indicating valid positions
            - "response_text": List of response text strings
            - "step": Current training step number
        tokenizer: Tokenizer instance for text processing
        gamma_name: Weight multiplier for tool name fields
        gamma_args: Weight multiplier for tool arguments
        alpha: Alpha parameter for length-dependent weighting  
        beta: Beta parameter for length-dependent weighting
        w_min: Minimum weight value
        w_max: Maximum weight value
        
    Returns:
        Token weight tensor of shape (B, T)
        
    Raises:
        ValueError: If batch format is invalid
        KeyError: If required keys are missing from batch
        TypeError: If input types are incorrect
    """

    # Extract current training step
    step = batch["step"]
    print(f"Current training step: {step}")
    if step is None:
        raise RuntimeError("No global_steps attribute found in _global_trainer, unable to get training step.")

    # Dynamic adjustment of format and think section weights
    # gamma_format = 0.8 - DEFAULT_STEP_WEIGHT_INCREASE * step  # High initially, decreases with step
    gamma_format = DEFAULT_GAMMA_FORMAT

    gamma_args = DEFAULT_BASE_THINK_WEIGHT + DEFAULT_STEP_WEIGHT_INCREASE * step  # Argument weight increases with step
    gamma_args = min(gamma_args, DEFAULT_MAX_GAMMA_ARGS)

    # Input validation
    if batch is None or not isinstance(batch, dict):
        raise ValueError(f"build_token_weight: input batch must be dict, got: {type(batch)}")
    if "loss_mask" not in batch:
        raise KeyError("build_token_weight: missing 'loss_mask'")
    if "response_text" not in batch:
        raise KeyError("build_token_weight: missing 'response_text'")
    if tokenizer is None:
        raise ValueError("build_token_weight: tokenizer cannot be None")

    loss_mask = batch["loss_mask"]  # (B, T)
    if not torch.is_tensor(loss_mask):
        raise TypeError(f"'loss_mask' must be torch.Tensor, got {type(loss_mask)}")
    B, T = loss_mask.shape

    response_texts = batch["response_text"]
    if not isinstance(response_texts, (list, tuple)) or len(response_texts) != B:
        raise ValueError("'response_text' must be List[str] with length equal to batch size")

    token_weight = torch.zeros_like(loss_mask, dtype=torch.float32)



    # Ensure gamma_args > gamma_name; auto-adjust if not satisfied
    A = float(gamma_args)
    Bv = float(gamma_name)

    for i in range(B):
        text = response_texts[i]
        if not isinstance(text, str):
            raise TypeError(f"response_text[{i}] must be str, got {type(text)}")

        # 1) Match and assign weights to format sections
        # 1a) <think>...</think> section - assign weight to tags and content
        _assign_tag_weights(text, THINK_TAG_PATTERN, token_weight, i, gamma_format, tokenizer)
        
        # Assign adaptive weight to <think> content
        for m in re.finditer(THINK_TAG_PATTERN, text, flags=re.S):
            content_lo, content_hi = m.span(1)
            c_t_lo, c_t_hi = char_span_to_token_span(text, content_lo, content_hi, tokenizer)
            if c_t_hi > c_t_lo:
                # Adaptive weight for think content based on training step
                gamma_think = DEFAULT_BASE_THINK_WEIGHT + DEFAULT_STEP_WEIGHT_INCREASE * step  # Increases with step
                gamma_think = min(gamma_think, DEFAULT_GAMMA_FORMAT)
                token_weight[i, c_t_lo:c_t_hi] = token_weight[i, c_t_lo:c_t_hi].maximum(
                    torch.full((c_t_hi - c_t_lo,), gamma_think, device=token_weight.device)
                )

        # 1b) <response>...</response> section - only assign weight to tags themselves
        _assign_tag_weights(text, RESPONSE_TAG_PATTERN, token_weight, i, gamma_format, tokenizer)

        # 1c) <tool_call>...</tool_call> section - only assign weight to tags themselves
        _assign_tag_weights(text, TOOL_CALL_TAG_PATTERN, token_weight, i, gamma_format, tokenizer)

        # 2) Match specific content in <tool_call> or <toolcall>
        for m in re.finditer(TOOL_CALL_CONTENT_PATTERN, text, flags=re.S):
            block = m.group(1)
            base_lo, base_hi = m.span(1)

            # 2a) "name": "..." - assign moderate weight to tool names
            nm = re.search(TOOL_NAME_PATTERN, block)
            if nm:
                c_lo = base_lo + nm.start(1)
                c_hi = base_lo + nm.end(1)
                t_lo, t_hi = char_span_to_token_span(text, c_lo, c_hi, tokenizer)
                if t_hi > t_lo:
                    token_weight[i, t_lo:t_hi] = token_weight[i, t_lo:t_hi].maximum(
                        torch.full((t_hi - t_lo,), Bv, device=token_weight.device)
                    )

            # 2b) "arguments"/"parameters": { ... } - assign highest weight to tool arguments
            km = re.search(TOOL_ARGS_PATTERN, block)
            if km:
                brace_pos = block.find('{', km.end())
                if brace_pos != -1:
                    loc_lo, loc_hi = _find_braced_span(block, brace_pos)
                    if loc_hi > loc_lo:
                        c_lo = base_lo + loc_lo
                        c_hi = base_lo + loc_hi
                        t_lo, t_hi = char_span_to_token_span(text, c_lo, c_hi, tokenizer)
                        if t_hi > t_lo:
                            token_weight[i, t_lo:t_hi] = token_weight[i, t_lo:t_hi].maximum(
                                torch.full((t_hi - t_lo,), A, device=token_weight.device)
                            )

        # 3) Clip only positive weights, then normalize on "positive weights ∩ valid positions" to mean=1
        pos_mask = (token_weight[i] > 0) & (loss_mask[i] != 0)
        if pos_mask.any():
            tw = token_weight[i][pos_mask].clamp(w_min, w_max)
            den = tw.mean().clamp_min(DEFAULT_NORMALIZATION_EPS)
            token_weight[i][pos_mask] = tw / den
        # 4) Keep only valid positions
        token_weight[i] *= (loss_mask[i] != 0).float()

    return token_weight
