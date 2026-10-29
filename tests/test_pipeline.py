import pytest
from src.normalizer import AttributeNormalizer
from src.matcher import BaselineMatcher, SemanticMatcher, MatchDecision


@pytest.fixture
def normalizer():
    return AttributeNormalizer()


@pytest.fixture
def baseline_matcher():
    return BaselineMatcher()


@pytest.fixture
def semantic_matcher():
    return SemanticMatcher()


def test_volume_normalization(normalizer):
    # 0.95 L and 32 oz should both normalize to approximately ~946-950 ml
    attr_liters = normalizer.normalize({"capacity": "0.95 L"})
    attr_oz = normalizer.normalize({"volume": "32 oz"})

    assert attr_liters.volume_ml is not None
    assert attr_oz.volume_ml is not None
    assert abs(attr_liters.volume_ml - attr_oz.volume_ml) < 10.0


def test_storage_normalization(normalizer):
    attr_128 = normalizer.normalize({"storage": "128 GB"})
    attr_tb = normalizer.normalize({"capacity": "1 TB"})

    assert attr_128.storage_gb == 128
    assert attr_tb.storage_gb == 1024


def test_color_extraction(normalizer):
    attr = normalizer.normalize({}, title="Apple iPhone 15 Pro (128 GB) - Natural Titanium")
    assert attr.color == "Natural Titanium"


def test_baseline_matcher_identical(baseline_matcher):
    title = "Sony WH-1000XM5 Wireless Headphones"
    sim = baseline_matcher.compute_similarity(title, title)
    assert sim >= 0.99


def test_semantic_matcher_variation_detection(semantic_matcher):
    item_black = {
        "title": "Sony WH-1000XM5 Headphones - Black",
        "raw_attributes": {"color": "Black"}
    }
    item_silver = {
        "title": "Sony WH-1000XM5 Headphones - Silver",
        "raw_attributes": {"color": "Silver"}
    }

    decision, similarity, metadata = semantic_matcher.classify_pair(item_black, item_silver)
    assert decision == MatchDecision.POTENTIAL_VARIATION
    assert any("color" in feat for feat in metadata["varying_features"])
