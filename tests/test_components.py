import pandas as pd
import pytest
from feature_extraction.extractor import IDSFeatureExtractor
from preprocessing.preprocessor import IDSPreprocessor

def test_feature_extractor():
    df = pd.DataFrame({'timestamp': [1], 'src_ip': ['1.1.1.1'], 'protocol': ['TCP'], 'length': [100]})
    extractor = IDSFeatureExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    assert 'length_log' in res.columns

def test_preprocessor():
    df = pd.DataFrame({'length_log': [1, 0, 1]})
    preprocessor = IDSPreprocessor()
    res = preprocessor.fit_transform(df)
    assert res.shape == df.shape
    assert abs(res['length_log'].mean()) < 0.1 # Standardized
