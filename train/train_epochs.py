import itertools

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm

from diffusers.training_utils import compute_snr
from train.checkpointing import save_checkpoint
from train.text_encoding import encode_prompt
from train.validate_and_test import log_validation


def train_epochs(accelerator, args, first_epoch, global_step,
                 train_dataloader, unet, vae, tokenizer, text_encoder,
                 noise_scheduler, optimizer, lr_scheduler, weight_dtype,
                 validation_prompt_embeds,
                 validation_negative_prompt_embeds,
                 logger, max_train_steps,
                 train_unet=True, train_te=True):
    if not train_unet and not train_te:
        raise Exception('no model to train')

    if train_unet and not train_te:
        name_prefix = 'u'
    elif train_te and not train_unet:
        name_prefix = 't'
    else:
        name_prefix = 's'
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc=f"{name_prefix}/Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train(train_unet)
        text_encoder.train(train_te)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                # Sample noise that we'll add to the model input
                if args.offset_noise:
                    noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                        model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                    )
                else:
                    noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                if args.pre_compute_text_embeddings:
                    encoder_hidden_states = batch["input_ids"]
                else:
                    encoder_hidden_states = encode_prompt(
                        text_encoder,
                        batch["input_ids"],
                        batch["attention_mask"],
                        text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                    )

                if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
                ).sample

                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Compute instance loss
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if train_unet and global_step % args.checkpointing_steps == 0:
                        save_checkpoint(args,
                                        accelerator,
                                        global_step,
                                        logger=logger)

                    if global_step % args.validation_steps == 0:
                        log_validation(
                            text_encoder,
                            tokenizer,
                            unet,
                            vae,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            validation_prompt_embeds,
                            validation_negative_prompt_embeds,
                            logger,
                            dirname_prefix=name_prefix
                        )

            if args.device_index is not None:
                k = 1024
                g = k * k * k
                # free, total = torch.cuda.mem_get_info(args.device_index)
                # occupied = total - free
                # occupied_gb = occupied / g
                reserved = torch.cuda.memory_reserved(args.device_index)
                reserved_gb = reserved / g
            else:
                # occupied_gb = 0.0
                reserved_gb = 0.0
            logs = {f"{name_prefix}-loss": loss.detach().item(),
                    f"{name_prefix}-lr": lr_scheduler.get_last_lr()[0],
                    # "gpu occupied": occupied_gb,
                    f"{name_prefix}-gpu-res": reserved_gb}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    return {'global_step': global_step}
