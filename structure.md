

```
data
    hf-accelerate
        xxx.yaml
    hf-pretrained
        darkSushiMixMix_225D
            feature_extractor
            safety_checker
            scheduler
            text_encoder
            tokenizer
            unet
            vae
    logs (tensorboard)
        hot
            t_{tid}
        cold
            t_{tid}
    notes
    sd_models
        controlnet-models
        controlnet-annotator-models
        models (sd checkpoints)
            adetailer
            Lora
            Stable-diffusion
            VAE
            ...
    sd-configs
        v1-inference.yaml
    trains
        t_{tid}
            instance_images
                1-xxx.jpg
            class_images
                1-xxx.png
            model
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
                    1-xxx.png
            test (images, final)
                1-xxx.png
```
