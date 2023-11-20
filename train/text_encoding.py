from transformers import PretrainedConfig
import torch


def import_text_encoder_class(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def pre_compute_text_embeddings(args, text_encoder, tokenizer):
    def do_compute(prompt):
        with torch.no_grad():
            text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
            prompt_embeds = encode_prompt(
                text_encoder,
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )

        return prompt_embeds

    pre_computed_encoder_hidden_states = do_compute(args.instance_prompt)
    validation_negative_prompt_embeds = do_compute("")

    if args.validation_prompt is not None:
        validation_prompt_encoder_hidden_states = do_compute(args.validation_prompt)
    else:
        validation_prompt_encoder_hidden_states = None

    if args.class_prompt is not None:
        class_prompt_encoder_hidden_states = do_compute(args.class_prompt)
    else:
        class_prompt_encoder_hidden_states = None

    return (pre_computed_encoder_hidden_states,
            validation_prompt_encoder_hidden_states,
            validation_negative_prompt_embeds,
            class_prompt_encoder_hidden_states)
