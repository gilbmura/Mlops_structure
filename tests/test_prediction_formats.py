"""The served HDF5 export preserves the trained model's complete prediction contract."""

import pytest

from config.settings import load_pipeline_config
from src.input_gate import FeatureDistance, GATE_FILE
from src.prediction import Predictor
from src.registry import KERAS_FILE, ModelRegistry
from tests.helpers import require_artifacts
from utils.files import read_csv


@pytest.mark.needs_artifacts
def test_h5_serving_matches_keras_predictions_and_screening(real_settings):
    require_artifacts(real_settings)
    registry = ModelRegistry(real_settings.models_dir)
    version = registry.active_version()
    config = load_pipeline_config()
    calibration = registry.read_attachment(version, GATE_FILE)
    gate = FeatureDistance.from_bytes(calibration) if calibration else None
    reference = Predictor(registry.load_model(version, KERAS_FILE), registry.metadata(version), config, gate)
    served = Predictor.from_registry(registry, config)

    assert served.artifact == "model.h5"
    # Spread across the class-sorted held-out split; this checks format parity, not model selection.
    for row in read_csv(real_settings.manifests_dir / "test.csv")[::55]:
        image = (real_settings.data_dir / row["path"]).read_bytes()
        assert served.predict_bytes(image).as_dict() == reference.predict_bytes(image).as_dict()
