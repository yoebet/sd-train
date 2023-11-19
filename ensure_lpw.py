from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                               custom_pipeline='lpw_stable_diffusion',
                                               local_files_only=True)
