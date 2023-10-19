

```
data
    logs
        hot (tensorboard)
            t_{tid}
        cold
            t_{tid}
    trains
        t_{tid}
            instance_images
            class_images
            output
                hf_proj (model)
                    feature_extractor
                    safety_checker
                    scheduler
                    text_encoder
                    tokenizer
                    unet
                    vae
                checkpoints
                    checkpoint-xxx
                validations
                    s_{step}
                test (images, final)
                logs
    training
        p_{pid}
    sd_models
        controlnet-models
        controlnet-annotator-models
        models (sd checkpoints)
            adetailer
            Lora
            Stable-diffusion
            VAE
            ...
    accelerate-configs
        xxx.yaml
```
