
from elasticsearch import Elasticsearch
from pathlib import Path
from eland.ml.pytorch import PyTorchModel
from eland.ml.pytorch.transformers import TransformerModel


def get_client_es():
    return Elasticsearch(
        hosts=[{'scheme': 'http', 'host': 'localhost', 'port': 9200}],
        request_timeout=300,
        verify_certs=False
    )


if __name__ == '__main__':
    tm = TransformerModel("distilbert-base-uncased-finetuned-sst-2-english", "text_classification")
    tmp_path = "models"
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    model_path, config, vocab_path = tm.save(tmp_path)
    #es = elasticsearch.Elasticsearch("http://localhost:9200", timeout=300, verify_certs=False)  # 5 minute timeout
    ptm = PyTorchModel(get_client_es(), tm.elasticsearch_model_id())
    ptm.import_model(model_path=model_path, config_path=None, vocab_path=vocab_path, config=config)
