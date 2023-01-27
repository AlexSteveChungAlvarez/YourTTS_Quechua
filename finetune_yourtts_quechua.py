import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
if __name__=="__main__":
    torch.set_num_threads(1)

    # pylint: disable=W0105
    """
        This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
        YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
        In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
        If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
        In addition, you will need to add the extra datasets following the VCTK as an example.
    """
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

    # Name of the run for the Trainer
    RUN_NAME = "YourTTS-QU-SINGLE"

    # Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
    OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

    # If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
    RESTORE_PATH = "..\..\AppData\Local\\tts\\tts_models--multilingual--multi-dataset--your_tts\model_file.pth"

    # This paramter is usefull to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
    SKIP_TRAIN_EPOCH = False

    # Set here the batch size to be used in training and evaluation
    BATCH_SIZE = 8

    # Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
    # Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
    SAMPLE_RATE = 16000

    # Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
    MAX_AUDIO_LEN_IN_SECONDS = 10

    # First run "preprocessing_quechua.py"
    DATASET_PATH = "datasets\QuechuaSingleSpeaker"

    # init configs
    quechua_config = BaseDatasetConfig(
        formatter="quechua",
        dataset_name="quechua_single_speaker",
        meta_file_train="transcripts.txt",
        meta_file_val="",
        path=DATASET_PATH,
        language="qu"
    )

    # Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to added new datasets just added they here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
    DATASETS_CONFIG_LIST = [quechua_config]

    ### Extract speaker embeddings
    SPEAKER_ENCODER_CHECKPOINT_PATH = (
        "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
    )
    SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

    D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

    # Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
    for dataset_conf in DATASETS_CONFIG_LIST:
        # Check if the embeddings weren't already computed, if not compute it
        embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
        if not os.path.isfile(embeddings_file):
            print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
            compute_embeddings(
                SPEAKER_ENCODER_CHECKPOINT_PATH,
                SPEAKER_ENCODER_CONFIG_PATH,
                embeddings_file,
                old_spakers_file=None,
                config_dataset_path=None,
                formatter_name=dataset_conf.formatter,
                dataset_name=dataset_conf.dataset_name,
                dataset_path=dataset_conf.path,
                meta_file_train=dataset_conf.meta_file_train,
                disable_cuda=False,
                no_eval=True,
            )
        D_VECTOR_FILES.append(embeddings_file)


    # Audio config used in training.
    audio_config = VitsAudioConfig(
        sample_rate=SAMPLE_RATE,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )

    # Init VITSArgs setting the arguments that is needed for the YourTTS model
    model_args = VitsArgs(
        d_vector_file=D_VECTOR_FILES,
        use_d_vector_file=True,
        d_vector_dim=512,
        num_layers_text_encoder=10,
        speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
        speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
        resblock_type_decoder="2",  # On the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
        # Usefull parameters to enable the Speaker Consistency Loss (SCL) discribed in the paper
        use_speaker_encoder_as_loss=True,
        # Usefull parameters to the enable multilingual training
        # use_language_embedding=True,
        # embedded_language_dim=4,
    )

    # General training config, here you can change the batch size and others usefull parameters
    config = VitsConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name="YourTTS_Quechua",
        run_description="""
                - Original YourTTS trained using Quechua single speaker dataset
            """,
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,
        eval_split_max_size=256,
        eval_split_size=0.014084507042253521,
        print_step=1000,
        plot_step=1000,
        log_model_step=1000,
        save_step=7000,
        save_n_checkpoints=5,
        save_checkpoints=True,
        target_loss="loss_1",
        print_eval=False,
        run_eval=False,
        use_phonemes=True,
        phonemizer="espeak",
        phoneme_language="qu",
        compute_input_seq_cache=True,
        add_blank=True,
        text_cleaner="multilingual_cleaners",
        phoneme_cache_path=os.path.join(OUT_PATH, "phoneme_cache"),
        precompute_num_workers=12,
        start_by_longest=True,
        datasets=DATASETS_CONFIG_LIST,
        cudnn_benchmark=False,
        max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
        mixed_precision=False,
        test_sentences=[
            [
                "Qantapuni qhawashayki warmiypaq.",
                "gui",
                None,
                "qu",
            ],
            [
                "Churiymi soqta wakata qatenqa orqoman.",
                "gui",
                None,
                "qu",
            ],
            [
                "Churiymi uyanta uphakun mayu unuwan.",
                "gui",
                None,
                "qu",
            ],
            [
                "Allin tuta kachun khunpa.",
                "gui",
                None,
                "qu",
            ],
            [
                "Wasiykita ripuy punkuyta qellichashanki.",
                "gui",
                None,
                "qu",
            ],
        ],
        # Enable the weighted sampler
        use_weighted_sampler=True,
        # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
        weighted_sampler_attrs={"speaker_name": 1.0},
        weighted_sampler_multipliers={},
        # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
        speaker_encoder_loss_alpha=9.0,
    )

    # Load all the datasets samples and split traning and evaluation sets
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Init the model
    model = Vits.init_from_config(config)

    # Init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()