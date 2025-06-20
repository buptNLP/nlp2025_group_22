# Step-by-step hard for mmlu_stem
sbs_hard = """Please reason step by step. 
Conclude with: 
Therefore, the final choice is: [answer].
Where [answer] is just the choice of the multiple choices that solves the problem."""

# Coarse-to-fine hard for mmlu_stem
c2f_hard = """Use the following pattern to solve the problem:
**Coarse-Grained Reasoning**
Provide a brief analysis of the problem, focusing on efficiency and conciseness.

**Fine-Grained Reasoning**
Provide detailed reasoning step by step of the problem, focusing on correctness and rigor.

Conclude with: 
Therefore, the final choice is: [answer].
Where [answer] is just the choice of the multiple choices that solves the problem."""

# Answer and Verify hard for mmlu_stem
aav_hard = """Use the following pattern to solve the problem:
**Quick Answer**
Provide an initial answer based on intuition or quick calculation.

**Verification**
Provide a revised answer through reasoning step by step. Correct previous mistakes, if any.

Conclude with: 
Therefore, the final choice is: [answer].
Where [answer] is just the choice of the multiple choices that solves the problem."""

# early stop and conclude before budget for mmlu_stem
early_stop = """\n\nNotice: When you are interrupted by the keyword **Time’s Up!**, stop reasoning immediately.
Based on your reasoning so far, conclude with: 
Therefore, the final choice is: [answer].
Where [answer] is just the choice of the multiple choices that solves the problem."""

# Step-by-step
sbs = sbs_hard + early_stop

# Coarse-to-fine
c2f = c2f_hard + early_stop

# Answer and Verify
aav = aav_hard + early_stop


# First define all unique chat templates
CHAT_TEMPLATE_FORMATS = {
    "mistral_format": "<s>[INST] {system_message}\n\n{input}[/INST]",
    
    "qwen_format": "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    
    "phi3mini_format": "<|system|>\n{system_message}<|end|>\n<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi3small_format": "<|endoftext|><|system|>\n{system_message}<|end|>\n<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi3medium_format": "<|user|>\n{input}<|end|>\n<|assistant|>\n",
    
    "phi4_format": "<|im_start|>system<|im_sep|>{system_message}<|im_end|><|im_start|>user<|im_sep|>{input}<|im_end|><|im_start|>assistant<|im_sep|>",
    
    "llama_format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    
    "gemma_format": "<bos><start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n",
    
    "deepseek-r1-distill_format" : "<｜begin▁of▁sentence｜><｜User｜>{input}<｜Assistant｜>"    # Avoid adding a system prompt; all instructions should be contained within the user prompt.
}


PROMPT_TEMPLATES = {
    "qwen-math-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-math-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-math-aav-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-math-sbs": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "qwen-math-c2f": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n",
    ),
    "qwen-math-aav": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n",
    ),
    "qwen-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-aav-hard": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "qwen-sbs": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "qwen-c2f": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),
    "qwen-aav": (
        CHAT_TEMPLATE_FORMATS["qwen_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    "mistral-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-aav-hard": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "mistral-sbs": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "mistral-c2f": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),
    "mistral-aav": (
        CHAT_TEMPLATE_FORMATS["mistral_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    "phi3mini-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", sbs_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", c2f_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", aav_hard),
        "{output}",
        "\n\n",
    ),
    "phi3mini-sbs": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "phi3mini-c2f": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),
    "phi3mini-aav": (
        CHAT_TEMPLATE_FORMATS["phi3mini_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    "phi3small-sbs": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", sbs),
        "{output}",
        "\n\n",
    ),
    "phi3small-c2f": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", c2f),
        "{output}",
        "\n\n", 
    ),  
    "phi3small-aav": (
        CHAT_TEMPLATE_FORMATS["phi3small_format"].replace("{system_message}", aav),
        "{output}",
        "\n\n", 
    ),
    "phi3medium-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", c2f_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-aav-hard": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-sbs": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", sbs + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "phi3medium-c2f": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", c2f + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    "phi3medium-aav": (
        CHAT_TEMPLATE_FORMATS["phi3medium_format"].replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n", 
    ),
    "deepseek-r1-distill-sbs-hard": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", sbs_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-c2f-hard": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", c2f_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-aav-hard": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", aav_hard + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-sbs": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", sbs + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-c2f": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", c2f + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    ),
    "deepseek-r1-distill-aav": (
        CHAT_TEMPLATE_FORMATS["deepseek-r1-distill_format"].replace("{input}", aav + "\n\n" + "{input}"),
        "{output}",
        "\n\n",
    )
}
