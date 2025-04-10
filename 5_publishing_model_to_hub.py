import numpy as np
import os
import pandas as pd
import datasets
from datasets import load_dataset, Audio, Dataset
from itertools import zip_longest
from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
from datasets import Dataset, Audio
import torch
import numpy as np
from pathlib import Path

model_name = "model_primstress_1e-5_20_1/checkpoint-6540/"
model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path=model_name
)

model.push_to_hub("5roop/Wav2Vec2BertPrimaryStressAudioFrameClassifier")
feature_extractor.push_to_hub("5roop/Wav2Vec2BertPrimaryStressAudioFrameClassifier")


exit()
