"""
ASIDE Model API Handler

This module provides the main API for working with ASIDE (Architecturally Separated 
Instruction-Data Embeddings) models and baseline approaches. It serves as the central
interface for model loading, inference, and evaluation across different embedding strategies.

Key Features:
- Unified API for multiple embedding types (vanilla, ISE, ASIDE)
- Support for major model families (Llama, Qwen, Mistral)
- Batch processing with efficient memory management
- Hidden state extraction for analysis
- Template-based prompt formatting
- DeepSpeed integration for large-scale inference

Architecture:
- CustomModelHandler: Main API class providing unified interface
- Model Registry: Automatic model class selection based on embedding type
- Batch Processing: Efficient inference with proper padding and attention handling
- Template System: Flexible prompt formatting for different model families

"""

import difflib
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

# import deepspeed
import einops
import openai
import torch
import torch.distributed as dist
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    pipeline,
)
import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

from model import *



def load_single_emb_model_and_tokenizer(
    model_name_or_path,
    tokenizer_path,
    model_dtype,
    chat_template_path=None,
    quant_4bit=False,
    model_cls=LlamaForCausalLM,
    config_cls=LlamaConfig,
    rotation_alpha=None,
    add_linear_shift=None,
    rotation_direction=None,
    learned_rotation=None,
    gradual_rotation=None,
):
    """
    Load model and tokenizer for single embedding approaches (vanilla, ISE, ASIDE).
    
    *Note: Original implementation involved having two physical embeddings. This is the only
    function used in the current implementation.
    
    This function handles loading of models that use a single token embedding matrix,
    as opposed to the legacy double embedding approach. It supports all ASIDE variants
    and baseline methods.
    
    Args:
        model_name_or_path (str): Path to model directory or HuggingFace model name
        tokenizer_path (str): Path to tokenizer (usually instruction-tuned version)
        model_dtype (torch.dtype): Data type for model weights (e.g., torch.bfloat16)
        chat_template_path (str, optional): Path to custom chat template file
        quant_4bit (bool): Whether to use 4-bit quantization for memory efficiency
        model_cls (class): Model class to instantiate (auto-selected based on embedding type)
        config_cls (class): Configuration class for the model
        rotation_alpha (float, optional): Rotation angle for ASIDE method (π/2 for 90°)
        add_linear_shift (bool, optional): Whether to add linear shift 
        rotation_direction (str, optional): Direction of rotation application ("right"/"left")  
        learned_rotation (bool, optional): Whether rotation is learned vs fixed
        gradual_rotation (bool, optional): Whether to apply rotation gradually during training
        
    Returns:
        tuple: (model, tokenizer) - Loaded and configured model and tokenizer
        
    Note:
        ASIDE-specific parameters (rotation_alpha, etc.) are stored in model config
        for use during forward passes. The rotation matrix is applied in the
        embedding layer based on token segment IDs.
    """
  
    print(
        f"CALLED load_vanilla_model_and_tokenizer on model {model_name_or_path} and tokenizer {tokenizer_path}"
    )

    config = config_cls.from_pretrained(model_name_or_path)

    if not hasattr(config, "rotation_alpha") or config.rotation_alpha is None:
        config.rotation_alpha = rotation_alpha
    if not hasattr(config, "add_linear_shift") or config.add_linear_shift is None:
        config.add_linear_shift = add_linear_shift
    if not hasattr(config, "rotation_direction") or config.rotation_direction is None:
        config.rotation_direction = rotation_direction
    if not hasattr(config, "learned_rotation") or config.learned_rotation is None:
        config.learned_rotation = learned_rotation
    if not hasattr(config, "gradual_rotation") or config.gradual_rotation is None:
        config.gradual_rotation = gradual_rotation
    print("Model config", config)
    if quant_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = model_cls.from_pretrained(  # AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=model_dtype,
        quantization_config=bnb_config if quant_4bit else None,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if chat_template_path is not None:
        with open(chat_template_path, "r") as f:
            chat_template = f.read()
            tokenizer.chat_template = chat_template

    tokenizer.padding_side = "left"

    if hasattr(model_cls, "_customize_tokenizer"):
        model_cls._customize_tokenizer(tokenizer, model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def get_target_modules(model):
    """
    Get all linear layers and embedding layers for LoRA targeting.
    
    Identifies modules suitable for LoRA fine-tuning by finding all linear
    and embedding layers in the model. Used when applying parameter-efficient
    fine-tuning methods.
    
    Args:
        model: LlamaModelForCausalLM or similar transformer model
        
    Returns:
        list: List of module names suitable for LoRA targeting
    """
    target_modules = []

    # Helper function to check if a module is a linear layer
    def is_linear_layer(module):
        return isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))

    # Helper function to check if a module is an embedding layer
    def is_embedding_layer(module):
        return isinstance(module, torch.nn.Embedding)

    # Iterate through all named modules
    for name, module in model.named_modules():
        if is_linear_layer(module):
            # Extract the final part of the module name
            module_name = name.split(".")[-1]
            if module_name not in target_modules:
                target_modules.append(module_name)
        elif is_embedding_layer(module) and "embed_tokens" in name:
            module_name = name.split(".")[-1]
            if module_name not in target_modules:
                target_modules.append(module_name)

    return target_modules


class CustomModelHandler:
    """
    Unified API handler for ASIDE models and baseline approaches.
    
    This class provides a consistent interface for working with different embedding
    strategies (vanilla, ISE, ASIDE) across multiple model families. It handles
    model loading, prompt formatting, batch processing, and response generation.
    
    Key Capabilities:
    - Automatic model class selection based on embedding type
    - Template-based prompt formatting for proper instruction-data separation  
    - Batch processing with memory-efficient padding
    - Hidden state extraction for analysis and probing
    - Integration with DeepSpeed for large-scale inference
    
    Embedding Types Supported:
    - 'single_emb': Vanilla transformer with single embedding matrix
    - 'double_emb': Legacy approach with separate embedding matrices (deprecated)
    - 'ise': ISE baseline with learnable offset vectors
    - 'forward_rot': ASIDE method with orthogonal rotations
    
    Model Families Supported:
    - Llama (2, 3.1)
    - Qwen 2.5  
    - Mistral 7B
    """
    def __init__(
        self,
        checkpoint_path: str,
        instruct_model_path: Optional[str],
        data_model_path: Optional[str],
        tokenizer_path: str,
        chat_template_path: Optional[str] = None,
        prompt_ix: int = 0,
        embedding_type: str = "single_emb",
        embeddings_init="copy",
        rotation_alpha=None,
        add_linear_shift=None,
        rotation_direction=None,
        learned_rotation=None,
        gradual_rotation=None,
        max_token_len=512,
        load_from_checkpoint=False,
        model_dtype=torch.bfloat16,
        rank=None,
        post_init_rotation=False,
    ) -> None:
        """
        Initialize the model handler with specified configuration.
        
        Args:
            checkpoint_path (str): Path to trained model checkpoint
            instruct_model_path (str, optional): Path to instruction model (for double_emb)
            data_model_path (str, optional): Path to data model (for double_emb) 
            tokenizer_path (str): Path to tokenizer (usually instruction-tuned)
            chat_template_path (str, optional): Path to custom chat template
            prompt_ix (int): Index of prompt template to use (default: 0)
            embedding_type (str): Type of embedding strategy
                - 'single_emb': Vanilla approach
                - 'double_emb': Legacy double embedding  
                - 'ise': ISE baseline
                - 'forward_rot': ASIDE method
            embeddings_init (str): Embedding initialization strategy
            rotation_alpha (float, optional): Rotation angle for ASIDE (π/2 = 1.57079633)
            add_linear_shift (bool, optional): Add linear shift for ISE
            rotation_direction (str, optional): Rotation direction ("right"/"left")
            learned_rotation (bool, optional): Whether rotation is learned
            gradual_rotation (bool, optional): Apply rotation gradually  
            max_token_len (int): Maximum token length for sequences
            load_from_checkpoint (bool): Whether to load from checkpoint vs base model
            model_dtype (torch.dtype): Model weight precision
            rank (int, optional): Distributed training rank
            post_init_rotation (bool): Apply rotation after initialization
        """
        assert embedding_type in ("single_emb", "double_emb", "ise", "forward_rot")
        if embedding_type == "single_emb":
            self.split_chat = False
        else:
            self.split_chat = True

        self.embedding_type = embedding_type
        self.instruct_model_path = instruct_model_path
        self.data_model_path = data_model_path
        self.tokenizer_path = tokenizer_path
        self.checkpoint_path = checkpoint_path
        self.chat_template_path = chat_template_path
        self.prompt_ix = prompt_ix
        self.model, self.tokenizer = None, None
        self.max_token_len = max_token_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_from_checkpoint = load_from_checkpoint
        self.model_dtype = model_dtype
        access_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.rank = rank
        self.embeddings_init = embeddings_init
        self.rotation_alpha = rotation_alpha
        self.add_linear_shift = add_linear_shift
        self.rotation_direction = rotation_direction
        self.learned_rotation = learned_rotation
        self.gradual_rotation = gradual_rotation
        self.post_init_rotation = post_init_rotation
        self.debug_printed = False
        if access_token:
            login(token=access_token)
        self._setup_hf_model()  # Stores Hugging Face models and tokenizers

    def get_template_parameters(self, template_type):
        """
        Get template-specific parameters for different model families.
        
        Returns the token lengths for different parts of the chat template,
        which are used for proper instruction-data token identification during
        hidden state extraction and analysis.
        
        Needed for interp. analysis.

        Args:
            template_type (str): Type of template ("base", "single", "ise", etc.)
            
        Returns:
            tuple: (system_prompt_len, template_infix_len, template_suffix_len)
                - system_prompt_len: Length of system prompt tokens
                - template_infix_len: Length of template infix (e.g., "Input:" tokens)
                - template_suffix_len: Length of template suffix tokens
                
        Note:
            These lengths are model-family specific and were determined empirically
            by analyzing the chat templates for each model family.
        """
        if "Llama-3.1-8B" in self.data_model_path:
            if template_type == "base":
                system_prompt_len = 30
                template_infix_len = 2
                template_suffix_len = 0
            elif template_type == "single":
                system_prompt_len = 55
                template_infix_len = 7
                template_suffix_len = 5
            elif template_type == "ise":
                system_prompt_len = 55
                template_infix_len = 8
                template_suffix_len = 5
            else:
                system_prompt_len = 55
                template_infix_len = 8
                template_suffix_len = 5
        elif "Llama-2-7b" in self.data_model_path:
            if template_type == "base":
                system_prompt_len = 30
                template_infix_len = 4
                template_suffix_len = 2
            elif template_type == "single":
                system_prompt_len = 48
                template_infix_len = 14
                template_suffix_len = 4
            elif template_type == "ise":
                system_prompt_len = 48
                template_infix_len = 14
                template_suffix_len = 4
            else:
                system_prompt_len = 48
                template_infix_len = 14
                template_suffix_len = 4
        elif "Llama-2-13b" in self.data_model_path:
            if template_type == "base":
                system_prompt_len = 30
                template_infix_len = 4
                template_suffix_len = 2
            elif template_type == "single":
                system_prompt_len = 48
                template_infix_len = 14
                template_suffix_len = 4
            elif template_type == "ise":
                system_prompt_len = 48
                template_infix_len = 14
                template_suffix_len = 4
            else:
                system_prompt_len = 48
                template_infix_len = 14
                template_suffix_len = 4
        elif "Qwen2.5-7B" in self.data_model_path:
            if template_type == "base":
                system_prompt_len = 32
                template_infix_len = 6
                template_suffix_len = 6
            elif template_type == "single":
                system_prompt_len = 32
                template_infix_len = 6
                template_suffix_len = 6
            elif template_type == "ise":
                system_prompt_len = 32
                template_infix_len = 6
                template_suffix_len = 6
            else:
                system_prompt_len = 32
                template_infix_len = 6
                template_suffix_len = 6
        elif "Mistral" in self.data_model_path:
            if template_type == "base":
                system_prompt_len = 37
                template_infix_len = 3
                template_suffix_len = 4
            elif template_type == "single":
                system_prompt_len = 39
                template_infix_len = 4
                template_suffix_len = 2
            elif template_type == "ise":
                system_prompt_len = 39
                template_infix_len = 6
                template_suffix_len = 2
            else:
                system_prompt_len = 39
                template_infix_len = 6
                template_suffix_len = 2
        else:
            raise ValueError(
                f"Template unknown for base model: {self.data_model_path} in arg self.data_model_path"
            )

        return system_prompt_len, template_infix_len, template_suffix_len

    def generate_one_token_with_attn(
        self, system_instruction: str, user_instruction: str
    ) -> Tuple[str, List[str], torch.Tensor, List[Tuple[str, str]]]:
        """
        Calls the appropriate model API based on the model family and formats the input accordingly.
        It generates exactly one token, and saves the attention weights during generateion.

        Parameters:
        - system_instruction (str): The system instruction for the model.
        - user_instruction (str): The user instruction for the model.

        Returns:
        - response: The model's response (one detokenized token)
        - input_str_tokens: a list of input tokens as strings
        - attn_patterns: The attention weights during generation, shape (num_layers, num_heads, dest_seq_len, src_seq_len).
        - model_input: the model's input
        """
        text_sequences = format_model_input(
            self.tokenizer,
            system_instruction,
            user_instruction,
            split_chat=self.embedding_type == "double_emb",
        )
        raise NotImplementedError("Legacy, must be adjusted to the new interface to be used.")
        if self.embedding_type == "double_emb":
            input_ids, attention_mask = texts_to_prepared_ids(
                text_sequences, self.tokenizer, self.max_token_len
            )  # , self.device)
        else:
            prompt = text_sequences[0][0]
            input_ids, attention_mask = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_token_len,
                truncation=True,
            ).values()
            print(input_ids[0][0])
            print(self.tokenizer.decode(input_ids[0][0]))

        if self.is_double_model:
            # Get token strings. Requires shifting tokens back that were shifted in `text_to_prepared_ids`, otherwise `convert_ids_to_tokens` gives None's.
            inp_ids = input_ids[0]
            shift_size = self.model.config.original_vocab_size
            data_tokens_mask = inp_ids >= shift_size
            transformed_inp_ids = inp_ids.detach().clone()
            transformed_inp_ids[data_tokens_mask] -= shift_size
            input_str_tokens = self.tokenizer.convert_ids_to_tokens(transformed_inp_ids)
            input_str_tokens = [token.replace("Ġ", " ") for token in input_str_tokens]
        else:
            pre_input_prompt = prompt.split(f"Input:")[0]
            pre_input_ids, pre_attention_mask = self.tokenizer(
                pre_input_prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_token_len,
                truncation=True,
            ).values()
            data_tokens_start = pre_input_ids.shape[1]
            data_tokens_mask = torch.zeros_like(pre_input_ids)
            data_tokens_mask[:, data_tokens_start:] = 1
            data_tokens_mask = data_tokens_mask.bool()
            input_str_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            input_str_tokens = [token.replace("Ġ", " ") for token in input_str_tokens]

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            input_length = input_ids.shape[1]  # ??

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,  # for deterministic generation
                num_beams=1,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
            output_sequences = outputs.sequences
            all_layer_attn = torch.stack(outputs.attentions[0])
            attn_patterns = einops.rearrange(
                all_layer_attn, "layer 1 head dest source -> layer head dest source"
            )

        # Decode and print the generated text
        output_texts = []
        for generated_sequence in output_sequences:
            text_output = self.tokenizer.decode(
                generated_sequence[input_length:] % len(self.tokenizer),
                skip_special_tokens=True,
            )

            output_texts.append(text_output)
        response = output_texts[0]

        return (
            response,
            input_str_tokens,
            data_tokens_mask,
            attn_patterns,
            text_sequences,
        )

    def get_probe_tokens(
        self, tokenizer: AutoTokenizer, full_string_tokens: List[str], probe_string: str
    ) -> Tuple[int, int]:
        """
        Locate probe tokens within the full token sequence.
        
        Finds the position of probe tokens (injection strings) within the data
        portion of the input. Used for analyzing how models process potential
        prompt injections embedded in data.
        
        Args:
            tokenizer: The model's tokenizer
            full_string_tokens: Complete list of string tokens
            probe_string: The probe/injection string to locate
            
        Returns:
            tuple: (start_index, end_index) of probe tokens in full sequence
            
        Note:
            Uses difflib for fuzzy matching to handle tokenization variations.
            The probe represents potential prompt injections in the data section.
        """
        probe_tokens = tokenizer.encode(probe_string, add_special_tokens=False)
        probe_tokens_str = tokenizer.convert_ids_to_tokens(probe_tokens)

        matcher = difflib.SequenceMatcher(None, full_string_tokens, probe_tokens_str)
        match = matcher.find_longest_match()

        return match.a, match.a + match.size

    def generate_one_token_with_hidden_states(
        self,
        system_instruction: str,
        user_instruction: str,
        system_prompt_len: int,
        template_infix_len,
        template_suffix_len: int,
        debug: bool = False,
        max_new_tokens: int = 1,
        probe_string: str = "",
        intervene_on_probe: bool = False,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[Tuple[str, str]]]:
        """
        Calls the appropriate model API based on the model family and formats the input accordingly.
        It generates exactly max new tokens, even though the name says otherwise and saves the hidden states during generateion.
        It saves hidden states of instruction and data tokens separately.

        Parameters:
        - system_instruction (str): The system instruction for the model.
        - user_instruction (str): The user instruction for the model.
        - system_prompt_len (int): The length of the system prompt.
        - max_new_tokens (int): The number of tokens to generate.
        - template_infix_len (int): The length of the template infix.
        - template_suffix_len (int): The length of the template suffix.
        - debug (bool): Whether to print debug information.
        - probe_string (str): A string that appears inside the data tokens that is also an instruction. The name `probe` comes from
            the original SEP paper. We use it to identify which tokens inside the prompt look like an instruction.
            This is used in our experiments to compare how the model activates instruction feature on this tokens, and also to
            intervene on them and let them go through instruction embeddings instead of data embeddings.



        Returns:
        - response: The model's response (one detokenized token)
        - instruction_tokens_str: the instruction tokens as strings
        - data_tokens_str: the data tokens as strings
        - probe_tokens_str: the probe tokens as strings
        - inst_hidden_states: hidden states of the instruction tokens, shape (num_inst_tokens, num_layers, hidden_size)
        - data_hidden_states: hidden states of the data tokens, shape (num_data_tokens, num_layers, hidden_size)
        - probe_hidden_states: hidden states of the probe tokens, shape (num_probe_tokens, num_layers, hidden_size)
        - last_tok_hidden_state: hidden state of the last input token, shape (num_layers, hidden_size)
        - text_sequences: the text sequences that were used to generate the response
        """
        text_sequences = format_model_input(
            self.tokenizer,
            system_instruction,
            user_instruction,
            split_chat=self.split_chat,
        )

        input_ids, attention_mask, segment_ids = texts_to_prepared_ids(
            text_sequences,
            self.tokenizer,
            self.max_token_len,
            model_type=self.embedding_type,
        )

        inst_tokens_start = system_prompt_len
        if self.embedding_type == "double_emb":
            double_inp_ids = input_ids[0, :-template_suffix_len]
            try:
                shift_size = self.model.config.original_vocab_size
            except AttributeError:
                # This works for ISE model
                shift_size = self.model.config.vocab_size

            data_tokens_mask = double_inp_ids >= shift_size
            data_tokens_start = (
                data_tokens_mask.shape[0] - data_tokens_mask.sum()
            ).item()
            # data_tokens_start += 2 # Skip the first two, since its "Input:"
        else:
            system_prompt_tokens = input_ids[0, :inst_tokens_start]
            prompt = self.tokenizer.decode(input_ids[0])
            # prompt = text_sequences[0][0]
            pre_input_prompt = prompt.split(f"Input:")[0]
            pre_input_ids, pre_attention_mask = self.tokenizer(
                pre_input_prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_token_len,
                truncation=True,
            ).values()
            data_tokens_start = pre_input_ids.shape[1]
            # data_tokens_start += 1 if self.embedding_type == "single" else 1 # Skip the "Input:", for some reason it's just 1 for ISE

        # Only for debug and visualization purposes.
        # Detach and clone are EXTREMELY important, otherwise you FUCK UP the model input!!!
        system_prompt_tokens = input_ids[0, :inst_tokens_start].detach().clone()
        instruction_tokens = (
            input_ids[0, inst_tokens_start : data_tokens_start - template_infix_len]
            .detach()
            .clone()
        )
        infix_tokens = (
            input_ids[0, data_tokens_start - template_infix_len : data_tokens_start]
            .detach()
            .clone()
        )
        template_suffix_start = input_ids.shape[1] - template_suffix_len
        data_tokens = (
            input_ids[0, data_tokens_start:template_suffix_start].detach().clone()
        )
        suffix_tokens = input_ids[0, template_suffix_start:].detach().clone()

        if self.embedding_type == "double_emb":
            for index in range(data_tokens.shape[0]):
                original_token = data_tokens[index]
                if original_token >= shift_size:
                    data_tokens[index] = original_token - shift_size

        instruction_tokens_str = self.tokenizer.convert_ids_to_tokens(
            instruction_tokens
        )
        data_tokens_str = self.tokenizer.convert_ids_to_tokens(data_tokens)

        probe_start, probe_end = self.get_probe_tokens(
            self.tokenizer, data_tokens_str, probe_string
        )
        abs_probe_start = data_tokens_start + probe_start
        abs_probe_end = data_tokens_start + probe_end
        probe_tokens = data_tokens[probe_start:probe_end]
        probe_tokens_str = data_tokens_str[probe_start:probe_end]

        if intervene_on_probe:
            # shift the probe tokens to the instruction tokens, but only for double model
            if self.embedding_type == "double_emb":
                for i in range(abs_probe_start, abs_probe_end):
                    input_ids[0, i] -= shift_size
            elif self.embedding_type == "forward_rot":
                for i in range(abs_probe_start, abs_probe_end):
                    segment_ids[0, i] = 0

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        if segment_ids is not None:
            segment_ids = segment_ids.to(self.model.device)

        with torch.no_grad():
            input_length = input_ids.shape[1]  # ??
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # for deterministic generation
                num_beams=1,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
            output_sequences = outputs.sequences
            all_layer_hidden_states = torch.stack(outputs.hidden_states[0])
            hidden_states = einops.rearrange(
                all_layer_hidden_states,
                "layer 1 seq_len hid_size -> seq_len layer hid_size",
            )
            # Apparently, hidden states are only taken for the input tokens, not for the generated tokens.
            assert hidden_states.shape[0] == input_length
            inst_hidden_states = hidden_states[
                inst_tokens_start : data_tokens_start - template_infix_len
            ]  # shape (num_inst_tokens, num_layers, hidden_size)
            data_hidden_states = hidden_states[
                data_tokens_start:template_suffix_start
            ]  # shape (num_data_tokens, num_layers, hidden_size)
            probe_hidden_states = hidden_states[
                abs_probe_start:abs_probe_end
            ]  # shape (num_probe_tokens, num_layers, hidden_size)
            last_tok_hidden_state = hidden_states[-1]  # shape (num_layers, hidden_size)

        # Decode and print the generated text
        output_texts = []
        for generated_sequence in output_sequences:
            text_output = self.tokenizer.decode(
                generated_sequence[input_length:] % len(self.tokenizer),
                skip_special_tokens=True,
            )

            output_texts.append(text_output)
        response = output_texts[0]

        if debug:
            raise ValueError("debug")

        return (
            response,
            instruction_tokens_str,
            data_tokens_str,
            probe_tokens_str,
            inst_hidden_states,
            data_hidden_states,
            probe_hidden_states,
            last_tok_hidden_state,
            text_sequences,
        )

    def generate_with_hidden_states_instruction_only(
        self, instruction_text: str, max_new_tokens: int = 1
    ) -> Tuple[str, torch.Tensor, List[Tuple[str, str]]]:
        """
        Generate `max_new_tokens` from the api, routing the prompt throught the instruction embeddings (for double model only ofcourse).
        Also save the hidden states of the instruction tokens.

        Parameters:
        - system_instruction (str): The system instruction for the model.

        Returns:
        - response: The model's response (one detokenized token)
        - inst_hidden_states: hidden states of the instruction tokens, shape (num_inst_tokens, num_layers, hidden_size)
        - model_input: the model's input
        """
        text_sequences = [(instruction_text, "inst")]
        raise NotImplementedError("Legacy, must be adjusted to the new interface to be used.")
        if self.is_double_model:
            input_ids, attention_mask = texts_to_prepared_ids(
                text_sequences, self.tokenizer, self.max_token_len
            )  # , self.device)
        else:
            prompt = text_sequences[0][0]
            input_ids, attention_mask = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_token_len,
                truncation=True,
            ).values()

        template_prefix_len = 1  # Because of addition of a 'bos' token, by both double, single and default models. Double has special, single and default use default bos token.

        instruction_tokens_str = self.tokenizer.convert_ids_to_tokens(
            input_ids[0, template_prefix_len:]
        )

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            input_length = input_ids.shape[1]  # ??

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # for deterministic generation
                num_beams=1,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
            output_sequences = outputs.sequences
            all_layer_hidden_states = torch.stack(outputs.hidden_states[0])
            hidden_states = einops.rearrange(
                all_layer_hidden_states,
                "layer 1 seq_len hid_size -> seq_len layer hid_size",
            )

            inst_hidden_states = hidden_states[
                template_prefix_len:
            ]  # shape (num_inst_tokens, num_layers, hidden_size), all hidden_states are for instruction here

        # Decode and print the generated text
        output_texts = []
        for generated_sequence in output_sequences:
            text_output = self.tokenizer.decode(
                generated_sequence[input_length:] % len(self.tokenizer),
                skip_special_tokens=True,
            )

            output_texts.append(text_output)
        response = output_texts[0]

        return response, instruction_tokens_str, inst_hidden_states, text_sequences

    def generate_with_hidden_states_data_only(
        self, data_text: str, max_new_tokens: int = 1
    ) -> Tuple[str, torch.Tensor, List[Tuple[str, str]]]:
        """
        Generate `max_new_tokens` from the api, routing the prompt throught the data embeddings (for double model only ofcourse).
        Also save the hidden states of the data tokens.

        Parameters:
        - system_instruction (str): The system instruction for the model.

        Returns:
        - response: The model's response (one detokenized token)
        - data_hidden_states: hidden states of the instruction tokens, shape (num_inst_tokens, num_layers, hidden_size)
        - model_input: the model's input
        """
        text_sequences = [(data_text, "data")]
        raise NotImplementedError("Legacy, must be adjusted to the new interface to be used.")
        # if self.is_double_model:
        #     input_ids, attention_mask = texts_to_prepared_ids(text_sequences, self.tokenizer,
        #                                                       self.max_token_len)  # , self.device)
        # else:
        #     prompt = text_sequences[0][0]
        #     input_ids, attention_mask = self.tokenizer(prompt,
        #                                                return_tensors="pt",
        #                                                padding='longest',
        #                                                max_length=self.max_token_len,
        #                                                truncation=True).values()

        # template_prefix_len = 1 # Because of addition of a 'bos' token, by both double, single and default models. Double has special, single and default use default bos token.

        # # CLONING IS VERY IMPORTANT
        # # Otherwise you fuck up the inputs to your model by the shift that comes next.
        # data_tokens = input_ids[0, template_prefix_len:].detach().clone()
        # shift_size = self.model.config.original_vocab_size
        # if self.is_double_model:
        #     data_tokens -= shift_size
        # data_tokens_str = self.tokenizer.convert_ids_to_tokens(data_tokens)

        # input_ids = input_ids.to(self.model.device)
        # attention_mask = attention_mask.to(self.model.device)

        # with torch.no_grad():
        #     input_length = input_ids.shape[1]  # ??

        #     outputs = self.model.generate(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         max_new_tokens=max_new_tokens,
        #         do_sample=False,  # for deterministic generation
        #         num_beams=1,
        #         top_p=None, temperature=None,
        #         pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
        #         bos_token_id=self.tokenizer.bos_token_id,
        #         output_hidden_sta
        # # Decode and print the generated text
        # output_texts = []
        # for generated_sequence in output_sequences:
        #     text_output = self.tokenizer.decode(
        #         generated_sequence[input_length:] % len(self.tokenizer), skip_special_tokens=True
        #     )

        #     output_texts.append(text_output)
        # response = output_texts[0]

        # return response, data_tokens_str,  data_hidden_states, text_sequences

    def prepare_batch_inputs(
        self,
        system_instructions: List[str],
        user_instructions: List[str],
    ) -> Dict[str, Union[torch.Tensor, List[int], int]]:
        """
        Prepare batched inputs with proper padding and segment IDs.
        
        This method handles the complex process of batching variable-length
        sequences while maintaining proper instruction-data separation through
        segment IDs and attention masks.
        
        Args:
            system_instructions: List of system/instruction prompts
            user_instructions: List of user/data inputs
            
        Returns:
            dict: Prepared batch data containing:
                - input_ids_batch: Padded token IDs (batch_size, max_seq_len)
                - attention_mask_batch: Attention masks (batch_size, max_seq_len)
                - segment_ids_batch: Segment routing IDs (batch_size, max_seq_len) or None
                - input_lengths: Original sequence lengths before padding
                - max_seq_len: Maximum sequence length after padding
                - model_inputs_for_logging: Formatted text sequences per example
                
        Note:
            Left-padding is used to ensure proper generation behavior.
            Segment IDs are only created for embedding types that require routing.
        """
        model_inputs_for_logging = []
        batch_size = len(system_instructions)
        all_input_ids = []
        all_segment_ids = []
        all_attention_masks = []
        input_lengths = []

        for sys_inst, usr_inst in zip(system_instructions, user_instructions):
            text_sequences = format_model_input(
                self.tokenizer,
                system_instruction=sys_inst,
                user_instruction=usr_inst,
                split_chat=self.split_chat,
            )
            model_inputs_for_logging.append(text_sequences)

            input_ids, attention_mask, segment_ids = texts_to_prepared_ids(
                text_sequences,
                self.tokenizer,
                max_length=self.max_token_len,
                model_type=self.embedding_type,
            )

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            input_lengths.append(input_ids.shape[1])
            if segment_ids is not None:
                all_segment_ids.append(segment_ids)

        max_seq_len = max(input_lengths)

        padded_input_ids = []
        padded_segment_ids = []
        padded_attention_masks = []

        for i in range(batch_size):
            seq_len = all_input_ids[i].shape[1]
            pad_len = max_seq_len - seq_len

            if pad_len > 0:
                pad_tensor = torch.full(
                    (1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long
                )
                padded_ids = torch.cat([pad_tensor, all_input_ids[i]], dim=1)

                pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
                padded_mask = torch.cat([pad_mask, all_attention_masks[i]], dim=1)

                if len(all_segment_ids):
                    pad_tensor_seg = torch.full((1, pad_len), 0, dtype=torch.long)
                    padded_seg_ids = torch.cat(
                        [pad_tensor_seg, all_segment_ids[i]], dim=1
                    )
            else:
                padded_ids = all_input_ids[i]
                padded_mask = all_attention_masks[i]
                if len(all_segment_ids):
                    padded_seg_ids = all_segment_ids[i]

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
            if len(all_segment_ids):
                padded_segment_ids.append(padded_seg_ids)

        input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.model.device)
        attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(
            self.model.device
        )
        segment_ids_batch = None
        if len(all_segment_ids):
            segment_ids_batch = torch.cat(padded_segment_ids, dim=0).to(
                self.model.device
            )

        return {
            "input_ids_batch": input_ids_batch,
            "attention_mask_batch": attention_mask_batch,
            "segment_ids_batch": segment_ids_batch,
            "input_lengths": input_lengths,
            "max_seq_len": max_seq_len,
            "model_inputs_for_logging": model_inputs_for_logging,
        }

    def generate_from_batch_inputs(
        self,
        prepared_inputs: Dict[str, Union[torch.Tensor, List[int], int]],
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate sequences from prepared batch inputs.
        
        Args:
            prepared_inputs: Dictionary from prepare_batch_inputs()
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling vs greedy decoding
            temperature: Sampling temperature (if do_sample=True)
            **generate_kwargs: Additional generation parameters
            
        Returns:
            torch.Tensor: Raw output sequences from model.generate()
        """
        input_ids_batch = prepared_inputs["input_ids_batch"]
        attention_mask_batch = prepared_inputs["attention_mask_batch"]
        segment_ids_batch = prepared_inputs.get(
            "segment_ids_batch"
        )  # Use .get for safety

        start_time = time.time()
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                segment_ids=segment_ids_batch,  # Pass None if not present
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=do_sample,
                num_beams=1,  # Assuming default from original code
                top_p=None,  # Assuming default from original code
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    128009,
                ],  # Keep original eos logic
                bos_token_id=self.tokenizer.bos_token_id,
                **generate_kwargs,  # Pass through any extra args
            )
        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.6f} seconds")
        return output_sequences

    def decode_batch_outputs(
        self,
        output_sequences: torch.Tensor,
        prepared_inputs: Dict[str, Union[torch.Tensor, List[int], int]],
    ) -> List[str]:
        """
        Decodes the raw output sequences from the model.

        Args:
            output_sequences: The tensor returned by generate_from_batch_inputs.
            prepared_inputs: The dictionary returned by prepare_batch_inputs (needed for lengths).

        Returns:
            A list of decoded response strings.
        """
        responses = []
        batch_size = output_sequences.shape[0]
        max_seq_len = prepared_inputs["max_seq_len"]  # Length after padding
        vocab_size_check = max(len(self.tokenizer), self.tokenizer.vocab_size)

        for i in range(batch_size):
            generated_ids = output_sequences[i]
            generated_text = self.tokenizer.decode(
                generated_ids[max_seq_len:] % vocab_size_check,
                skip_special_tokens=True,
            )
            responses.append(generated_text)
        return responses

    def call_model_api_batch_intervenable(
        self,
        system_instructions: List[str],
        user_instructions: List[str],
        max_new_tokens=1024,
        do_sample=False,
        temperature=None,
        return_prepared_inputs=False,
        **generate_kwargs,  # Allow passing extra kwargs
    ):
        """
        Batch API call with intervention capability.
        
        This method exposes the intermediate input preparation step, allowing
        for interventions on the prepared inputs before generation. Useful for
        controlled experiments and analysis.
        
        Args:
            system_instructions: List of system/instruction prompts
            user_instructions: List of user/data inputs
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            return_prepared_inputs: If True, return prepared inputs without generation
            **generate_kwargs: Additional generation parameters
            
        Returns:
            If return_prepared_inputs=True: prepared_inputs dict
            Otherwise: (responses, model_inputs_for_logging) tuple
        """
        # 1. Prepare Inputs
        prepared_inputs = self.prepare_batch_inputs(
            system_instructions=system_instructions,
            user_instructions=user_instructions,
        )

        if return_prepared_inputs:
            return prepared_inputs

        output_sequences = self.generate_from_batch_inputs(
            prepared_inputs=prepared_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **generate_kwargs,
        )

        # 3. Decode Outputs
        responses = self.decode_batch_outputs(
            output_sequences=output_sequences, prepared_inputs=prepared_inputs
        )

        return responses, prepared_inputs["model_inputs_for_logging"]

    def call_model_api_batch(
        self,
        system_instructions: List[str],
        user_instructions: List[str],
        max_new_tokens=1024,
        do_sample=False,
        temperature=None,
        top_p=None
    ):
        """
        Main batch inference API for efficient model evaluation.
        
        This is the primary method for generating responses in batch mode,
        handling proper instruction-data separation based on embedding type.
        
        Args:
            system_instructions: List of system/instruction prompts
            user_instructions: List of user/data inputs  
            max_new_tokens: Maximum tokens to generate per example
            do_sample: Whether to use sampling vs greedy decoding
            temperature: Sampling temperature (if do_sample=True)
            
        Returns:
            tuple: (responses, model_inputs_for_logging)
                - responses: List of generated text strings
                - model_inputs_for_logging: List of formatted input sequences
                
        Workflow:
            1. Format inputs according to embedding type requirements
            2. Tokenize and pad sequences with appropriate segment routing
            3. Generate responses using efficient batch processing
            4. Decode and return results
            
        """
        # For debugging or logging, store each example's model input

        model_inputs_for_logging = []

        # If we have N items in the batch:
        batch_size = len(system_instructions)

        # We'll collect each example's tokenized results in lists
        all_input_ids = []
        all_segment_ids = []
        all_attention_masks = []
        input_lengths = []

        i = 0
        for sys_inst, usr_inst in zip(system_instructions, user_instructions):
            # Format the input (chat) according to your custom logic:
            text_sequences = format_model_input(
                self.tokenizer,
                system_instruction=sys_inst,
                user_instruction=usr_inst,
                split_chat=self.split_chat,
            )
            model_inputs_for_logging.append(text_sequences)

            input_ids, attention_mask, segment_ids = texts_to_prepared_ids(
                text_sequences,
                self.tokenizer,
                max_length=self.max_token_len,
                model_type=self.embedding_type,
            )

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            input_lengths.append(input_ids.shape[1])
            if segment_ids is not None:
                all_segment_ids.append(segment_ids)

        # --------------------------------------------------------------------
        # Pad all examples in this batch to the same max sequence length
        # --------------------------------------------------------------------
        max_seq_len = max(input_lengths)

        padded_input_ids = []
        padded_segment_ids = []
        padded_attention_masks = []

        for i in range(batch_size):
            seq_len = all_input_ids[i].shape[1]

            # How many tokens to pad to match max_seq_len
            pad_len = max_seq_len - seq_len

            # Pad input_ids
            if pad_len > 0:
                pad_tensor = torch.full(
                    (1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long
                )
                padded_ids = torch.cat([pad_tensor, all_input_ids[i]], dim=1)
            else:
                padded_ids = all_input_ids[i]
            if len(all_segment_ids):
                if pad_len > 0:
                    pad_tensor = torch.full((1, pad_len), 0, dtype=torch.long)
                    padded_seg_ids = torch.cat([pad_tensor, all_segment_ids[i]], dim=1)
                else:
                    padded_seg_ids = all_segment_ids[i]

            # Pad attention_mask
            if pad_len > 0:
                pad_mask = torch.zeros((1, pad_len), dtype=torch.long)
                padded_mask = torch.cat([pad_mask, all_attention_masks[i]], dim=1)
            else:
                padded_mask = all_attention_masks[i]

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
            if len(all_segment_ids):
                padded_segment_ids.append(padded_seg_ids)

        # Concatenate into a single batch
        input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.model.device)
        if len(all_segment_ids):
            segment_ids_batch = torch.cat(padded_segment_ids, dim=0).to(
                self.model.device
            )
        else:
            segment_ids_batch = None
        attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(
            self.model.device
        )
        # if not self.debug_printed:
        #     self.debug_printed = True
        #     print("sys instr", system_instructions[0])
        #     print("user instr", user_instructions[0])
        #     print("INPUT IDS BATCH\n", input_ids_batch[0])
        #     if segment_ids_batch is not None:
        #         print("SEGMENT IDS BATCH", segment_ids_batch[0])
        #     print("ATTN BATCH", attention_mask_batch[0])

        #     print("=== DeepSpeed Engine Debug Info ===")

        #     some_param = next(self.model.parameters())
        #     device = some_param.device
        #     input_ids_batch = input_ids_batch.to(device)
        #     attention_mask_batch = attention_mask_batch.to(device)
        #     print("Sample param dtype:", some_param.dtype)
        #     print("Sample param device:", some_param.device)
        #     print("input_ids dtype:", input_ids_batch.dtype)
        #     print("input_ids device:", input_ids_batch.device)
        #     print("attention_mask dtype:", attention_mask_batch.dtype)
        #     print("attention_mask device:", attention_mask_batch.device)

        #     print(
        #         f"Shapes, L710: {input_ids_batch.shape, attention_mask_batch.shape, segment_ids_batch.shape if segment_ids_batch is not None else None}"
        #     )
        start_time = time.time()

        with torch.no_grad():
            if self.embedding_type == "single_emb":
                output_sequences = self.model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=do_sample,  # for deterministic generation
                    num_beams=1,
                    top_p=top_p,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.eos_token_id, 128009],
                    bos_token_id=self.tokenizer.bos_token_id,
                )
            else:
                output_sequences = self.model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    segment_ids=segment_ids_batch if len(all_segment_ids) else None,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=do_sample,  # for deterministic generation
                    num_beams=1,
                    top_p=top_p,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.eos_token_id, 128009],
                    bos_token_id=self.tokenizer.bos_token_id,
                )

        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.6f} seconds")
        # --------------------------------------------------------------------
        # Decode each example's output
        # --------------------------------------------------------------------
        # We'll store the textual responses in this list, one per example
        responses = []

        # Because each example can have a different input length, we decode carefully.
        for i in range(batch_size):
            seq_len = input_lengths[i]  # The "un-padded" portion for example i
            # output_sequences[i] is a 1D tensor of token IDs for example i
            generated_ids = output_sequences[i]

            # The newly generated tokens come after the original input tokens.
            # We skip the first `seq_len` tokens to get just the model's new output.
            generated_text = self.tokenizer.decode(
                generated_ids[max_seq_len:]
                % max(len(self.tokenizer), self.tokenizer.vocab_size),
                skip_special_tokens=True,
            )
            responses.append(generated_text)

        return responses, model_inputs_for_logging

    def _setup_hf_model(self) -> None:
        """
        Initialize the HuggingFace model and tokenizer with proper configuration.
        
        This method uses a registry pattern to automatically select the appropriate
        model and configuration classes based on the model family and embedding type.
        It handles the complexity of different model architectures transparently.
        
        Model Registry Structure:
        - CONFIG_CLASS_REGISTRY: Maps model families to configuration classes
        - MODEL_CLASS_REGISTRY: Maps (model_family, embedding_type) to model classes
        
        Supported Combinations:
        - Llama + (single_emb, double_emb, ise, forward_rot)
        - Qwen + (single_emb, ise, forward_rot)  
        - Mistral + (single_emb, ise, forward_rot)
        
        Note:
            Double embedding is only supported for Llama models and is deprecated.
            New experiments should use forward_rot (ASIDE) or ise baselines.
        """

        CONFIG_CLASS_REGISTRY = {
            "llama": CustomLlamaConfig,
            "qwen": CustomQwenConfig,
            "mistral": CustomMistralConfig,
        }

        MODEL_CLASS_REGISTRY = {
            "llama": {
                "double_emb": CustomLLaMA,
                "single_emb": LlamaForCausalLM,
                "ise": LlamaISE,
                "forward_rot": LlamaForwardRot,
            },
            "qwen": {
                "single_emb": Qwen2ForCausalLM,
                "ise": QwenISE,
                "forward_rot": QwenForwardRot,
            },
            "mistral": {
                "single_emb": MistralBase,
                "ise": MistralISE,
                "forward_rot": MistralForwardRot,
            }
        }

        model_name = None
        for model in MODEL_CLASS_REGISTRY.keys():
            if model in self.tokenizer_path.lower():
                model_name = model
        if model_name is None:
            raise ValueError("Unknown model")

        config_cls = CONFIG_CLASS_REGISTRY.get(model_name)
        model_cls = MODEL_CLASS_REGISTRY[model_name].get(self.embedding_type)
        if config_cls is None or model_cls is None:
            raise ValueError("Unsupported (model, embed_style)")

        print("\n", config_cls, model_cls, "\n")

        if self.embedding_type == "double_emb":
            if model_name == "llama":
                self.model, self.tokenizer = model_cls.setup_model_and_tok(
                    self.checkpoint_path,
                    self.instruct_model_path,
                    self.data_model_path,
                    self.tokenizer_path,
                    embedding_init=self.embeddings_init,
                    rotation_alpha=self.rotation_alpha,
                    device="cpu",
                    load_from_checkpoint=self.load_from_checkpoint,
                    model_dtype=self.model_dtype,
                    post_init_rotation=self.post_init_rotation,
                )
            else:
                raise ValueError("Double embed for other models unsupported")
        else:
            self.model, self.tokenizer = load_single_emb_model_and_tokenizer(
                self.checkpoint_path
                if self.load_from_checkpoint
                else self.instruct_model_path,
                self.tokenizer_path,
                self.model_dtype,
                model_cls=model_cls,
                config_cls=config_cls,
                rotation_alpha=self.rotation_alpha,
                add_linear_shift=self.add_linear_shift,
                rotation_direction=self.rotation_direction,
                learned_rotation=self.learned_rotation,
                gradual_rotation=self.gradual_rotation,
            )
        print(f"chat_template_path: {self.chat_template_path}")
        print("\n", "MODEL TYPE: ", type(self.model))

        if self.chat_template_path is not None:
            with open(self.chat_template_path, "r") as f:
                chat_template = f.read()
                self.tokenizer.chat_template = chat_template


def format_model_input(
    tokenizer,
    system_instruction: str,
    user_instruction: str,
    assistant_message: str = None,
    split_chat=False,
) -> List[Tuple[str, str]]:
    """
    Format model input according to embedding type requirements.
    
    This function handles the critical task of formatting prompts for different
    embedding strategies. For ASIDE/ISE models, it splits the chat at the 
    "Input:" separator to enable proper instruction-data routing.
    
    Args:
        tokenizer: The model's tokenizer (with chat template)
        system_instruction (str): The system/instruction prompt
        user_instruction (str): The user/data input
        assistant_message (str, optional): Pre-filled assistant response
        split_chat (bool): Whether to split for instruction-data separation
        
    Returns:
        list: List of (text, role) tuples for tokenization
            - For vanilla models: [("full_chat", "inst")]
            - For ASIDE/ISE models: [("instruction_part", "inst"), ("data_part", "data")]
            
    Chat Template Processing:
        1. Apply tokenizer's chat template if available
        2. Otherwise use simple concatenation format  
        3. Split at "Input:" separator if split_chat=True
        4. Assign appropriate routing roles ("inst" vs "data")
        
    Note:
        The split_chat parameter is automatically set based on embedding type
        in CustomModelHandler initialization. This ensures proper routing for
        instruction-data separation methods.
    """
  
    if tokenizer.chat_template is not None:
        chat = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction},
        ]
        if assistant_message is not None:
            chat.append({"role": "assistant", "content": assistant_message})

        chat = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=assistant_message is None
        )
    else:
        chat = system_instruction + "\n" + user_instruction + "\n"
        if assistant_message is not None:
            chat += assistant_message
        else:
            chat += "Response:"
    sep_sequence = "Input:\n"
    sep_sequence_start = chat.find(sep_sequence)
    if split_chat:
        chat_pieces = [
            chat[:sep_sequence_start],
            chat[sep_sequence_start:],
        ]  # chat.split(SEP_TOKEN)

        text_sequences = [
            (chat_pieces[i], "inst" if i % 2 == 0 else "data")
            for i in range(len(chat_pieces))
        ]

    else:
        text_sequences = [(chat, "inst")]
    return text_sequences  # [(Do me this, "inst"), (2+2, "data")]


def load_config(config_path: str = "./config.json") -> Dict:
    """
    Loads configuration settings from a JSON file.

    Parameters:
    - config_path (str): The path to the configuration JSON file.

    Returns:
    - Dict: The loaded configuration settings.
    """
    with open(
        config_path,
        "r",
    ) as file:
        return json.load(file)


def load_data(
    data_path: str, templates_path: str, prompt_index: int
) -> Tuple[List[Dict], Dict]:
    """
    Load evaluation dataset and prompt templates.
    
    Args:
        data_path (str): Path to dataset JSON file
        templates_path (str): Path to prompt templates JSON file  
        prompt_index (int): Index of prompt template to use
        
    Returns:
        tuple: (dataset, selected_prompt_template)
    """
    with open(data_path, "r") as f:
        dataset = json.load(f)
    with open(templates_path, "r") as f:
        prompt_template = json.load(f)[prompt_index]
    return dataset, prompt_template


def prepare_for_formatting(s: str) -> str:
    """
    Prepare string for Python format() by escaping braces.
    
    Args:
        s (str): String potentially containing format braces
        
    Returns:
        str: String with braces properly escaped for formatting
    """
    border = s.find("}")
    new_s = s[: border + 1] + s[border + 1 :].replace("}", "}}").replace("{", "{{")
    return new_s


def format_prompt(prompt, template, role):
    """
    Format a prompt using template for specific role.
    
    Args:
        prompt: Prompt text to format
        template: Template dictionary with role-specific formats
        role: Role type ("system", "user", etc.)
        
    Returns:
        str: Formatted prompt text
    """
    if role == "user" and len(prompt) < 2:
        prompt = "No input"
    return prepare_for_formatting(template[role]).format(prompt)


def format_prompt_for_sep_inference(
    elem: Dict, template: Dict, mode: str = "data_with_probe"
) -> Tuple[str, str]:
    """
    Format prompts for SEP (instruction-data separation) evaluation.
    
    This function creates the two prompt variants used in SEP evaluation:
    1. Probe in data section (should NOT be executed)
    2. Probe in instruction section (should be executed)
    
    Args:
        elem (Dict): SEP dataset element containing prompts and probes
        template (Dict): Prompt template for formatting
        mode (str): Formatting mode
            - 'data_with_probe': Probe appears in data section
            - 'probe_with_task': Probe appears in instruction section
            
    Returns:
        tuple: (system_instruction, user_instruction) formatted for evaluation
        
    Note:
        This is used for the core ASIDE evaluation metric. The model should
        execute probes only when they appear in instruction context, not data context.
    """

    if mode == "data_with_probe":
        system_instruction = format_prompt(
            elem["system_prompt_clean"],
            template,
            "system",
        )  # no need to add sep in system
        user_instruction = format_prompt(
            elem["prompt_instructed"],
            template,
            "user",
        )
    elif mode == "probe_with_task":
        system_instruction = format_prompt(
            elem["system_prompt_instructed"],
            template,
            "system",
        )  # no need to add sep in system
        user_instruction = format_prompt(elem["prompt_clean"], template, "user")
    else:
        raise ValueError(
            f"Invalid mode for prompt formatting: {mode}. Valid modes are 'data_with_probe' or 'probe_with_task'."
        )
    return system_instruction, user_instruction



def inference(
    dataset: List[Dict],
    output_path: str,
    template_info: Dict,
    handler: CustomModelHandler,
    save_step: int = 2,
    batch_size: int = 8,  # <--- new!
    mp_size: Optional[int] = None,
) -> None:
    
    """
    Run batched inference for SEP evaluation with DeepSpeed optimization.
    
    This function performs the core ASIDE evaluation by running models on the
    SEP dataset with both prompt variants (probe in data vs instruction) and
    comparing the outputs to measure instruction-data separation.
    
    Args:
        dataset: List of SEP evaluation examples
        output_path: Path to save inference results JSON
        template_info: Template configuration information
        handler: Initialized CustomModelHandler for the model
        save_step: Save results every N steps (for progress checkpointing)
        batch_size: Number of examples to process per batch
        
    Workflow:
        1. Initialize DeepSpeed inference engine for efficient processing
        2. Process dataset in batches to manage memory usage
        3. Generate responses for both prompt variants per example
        4. Save results in SEP evaluation format
        
    Output Format:
        Each result contains:
        - output1_probe_in_data: Response when probe is in data section
        - output2_probe_in_task: Response when probe is in instruction section  
        - checkpoint: Model checkpoint path
        - instructions: Formatted input sequences
        - data: Original SEP dataset element
        
    Note:
        Uses DeepSpeed for memory-efficient inference on large models.
        The double_emb models require special checkpoint loading handling.
    """
    output = []
    # handler.model.to("cuda") # USE IF NOT DEEPSPEED

    # handler.model = handler.model.half().to("cuda")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if handler.embedding_type == "double_emb":
        handler.model = load_state_dict_from_zero_checkpoint(
            handler.model, handler.checkpoint_path
        )


    engine = deepspeed.init_inference(
        model=handler.model,
        mp_size=torch.cuda.device_count() if mp_size is None else mp_size,  # e.g., 2 or 4 # SHOULD BE cuda num of devices
        dtype=torch.bfloat16,  # or another precision
        replace_method="auto",
        replace_with_kernel_inject=False,  # False
    )
    handler.model = engine.module
    handler.model.eval()
    handler.model.config.use_cache = True

    for start_idx in tqdm(range(0, len(dataset), batch_size)):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_data = dataset[start_idx:end_idx]

        sys_instr_data_list = []
        user_instr_data_list = []
        sys_instr_task_list = []
        user_instr_task_list = []

        # Collect prompts for the entire batch
        for data_point in batch_data:
            sys_instr_1, user_instr_1 = format_prompt_for_sep_inference(
                data_point, template_info["template_prompt"], mode="data_with_probe"
            )
            sys_instr_2, user_instr_2 = format_prompt_for_sep_inference(
                data_point, template_info["template_prompt"], mode="probe_with_task"
            )
            sys_instr_data_list.append(sys_instr_1)
            user_instr_data_list.append(user_instr_1)
            sys_instr_task_list.append(sys_instr_2)
            user_instr_task_list.append(user_instr_2)

        # Call the model in batches
        response_data_list, input_data_list = handler.call_model_api_batch(
            sys_instr_data_list, user_instr_data_list
        )
        response_task_list, input_task_list = handler.call_model_api_batch(
            sys_instr_task_list, user_instr_task_list
        )

        # Map the responses back to the data points
        for i, data_point in enumerate(batch_data):
            response1 = response_data_list[i]
            response2 = response_task_list[i]
            data_point.update(template_info)

            output.append(
                {
                    "output1_probe_in_data": response1,
                    "output2_probe_in_task": response2,
                    "checkpoint": handler.checkpoint_path,
                    "instructions": {
                        "input_1": input_data_list[i],
                        "input_2": input_task_list[i],
                    },
                    "data": data_point,
                }
            )

        # Periodically save partial results
        for ix in range(start_idx, end_idx):
            if ix % save_step == 0 or end_idx == len(dataset):
                with open(output_path, "w+") as f:
                    json.dump(output, f, indent=2)  # optional indent for readability
            break

    with open(output_path, "w+") as f:
        json.dump(output, f, indent=2)
