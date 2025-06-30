from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

class TIEDModelConfig(PretrainedConfig):
    model_type = "TIED"
    is_composition = True

    def __init__(
        self,
        text_encoder_config=None,
        text_encoder_model=None,
        vae_model=None,
        vae_config=None,
        decoder_config=None,
        image_size=256,
        hidden_size=1024,
        vocab_size=1,
        z_step=32,
        text_prompt_pooling_type="first",
        projector_hidden_act="gelu",
        reduction = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_encoder_config, dict):
            model_type = text_encoder_config.get("model_type", "bert")
            text_encoder_config = CONFIG_MAPPING[model_type](**text_encoder_config)
        elif text_encoder_config is not None and not isinstance(text_encoder_config, PretrainedConfig):
            raise ValueError("text_encoder_config must be dict or PretrainedConfig")

        if isinstance(decoder_config, dict):
            model_type = decoder_config.get("model_type", "llama")
            decoder_config = CONFIG_MAPPING[model_type](**decoder_config)
        elif decoder_config is not None and not isinstance(decoder_config, PretrainedConfig):
            raise ValueError("decoder_config must be dict or PretrainedConfig")

        self.reduction = reduction
        self.z_step = z_step
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.text_encoder_config = text_encoder_config
        self.text_encoder_model = text_encoder_model
        self.vae_model = vae_model
        self.vae_config = vae_config
        self.decoder_config = decoder_config
        self.vocab_size = vocab_size
        self.text_prompt_pooling_type = text_prompt_pooling_type
        self.projector_hidden_act = projector_hidden_act

    def to_dict(self):
        output = super().to_dict()
        output["text_encoder_config"] = (
            self.text_encoder_config.to_dict() if self.text_encoder_config else None
        )
        output["decoder_config"] = (
            self.decoder_config.to_dict() if self.decoder_config else None
        )
        output["vae_config"] = dict(self.vae_config) if self.vae_config else None
        output["text_encoder_model"] = self.text_encoder_model
        output["vae_model"] = self.vae_model
        return output
