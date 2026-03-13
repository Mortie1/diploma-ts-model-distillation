class NullWriter:
    """No-op writer for local and CI runs."""

    def __init__(self, logger, project_config, run_name="local", loss_names=None, **kwargs):
        self.step = 0
        self.mode = ""
        self.run_name = run_name
        self.loss_names = loss_names or ["loss"]
        self.log_checkpoints = kwargs.get("log_checkpoints", False)

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def add_checkpoint(self, checkpoint_path, save_dir):
        return None

    def add_scalar(self, scalar_name, scalar):
        return None

    def add_scalars(self, scalars):
        return None

    def add_image(self, image_name, image):
        return None

    def add_audio(self, audio_name, audio, sample_rate=None):
        return None

    def add_text(self, text_name, text):
        return None

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        return None

    def add_table(self, table_name, table):
        return None
