from models.multimodal_model import MultimodalModel
from processors.ecg_processor import ECGProcessor
from processors.metadata_processor import MetadataProcessor
from processors.text_processor import TextProcessor

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


class ApplicationController:
    def __init__(self):
        self.model = MultimodalModel(
            binary_path="ecg_model_hierarchical/model_binary_v6",
            abn_path="ecg_model_hierarchical/model_abn_v6"
        )
        self.ecg_proc  = ECGProcessor()
        self.meta_proc = MetadataProcessor()
        self.text_proc = TextProcessor()

    def predict(self, ecg_file=None, metadata=None, text=""):
        ecg  = self.ecg_proc.process(ecg_file)
        meta = self.meta_proc.process(metadata, ecg, text)
        txt  = self.text_proc.process(text)
        return self.model.predict(ecg, meta, txt)