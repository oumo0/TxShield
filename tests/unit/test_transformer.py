from txshield.detection.features.transformer import transform_feature_row

def test_transformer_outputs_ten_features():
    assert transform_feature_row([0.0] * 14).shape == (10,)
