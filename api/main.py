import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.matcher import SemanticMatcher, MatchDecision
from src.normalizer import AttributeNormalizer, NormalizedAttributes

app = FastAPI(
    title="Amazon Catalog Deduplication & Attribute Normalization Service",
    description="High-throughput API for resolving duplicate listings and standardizing e-commerce catalog attributes.",
    version="1.0.0"
)

# Instantiate models at service startup
matcher = SemanticMatcher()
normalizer = AttributeNormalizer()


class ProductItem(BaseModel):
    title: str = Field(..., example="Sony WH-1000XM5 Wireless Noise Canceling Headphones - Black")
    brand: Optional[str] = Field(None, example="Sony")
    category: Optional[str] = Field(None, example="Electronics")
    raw_attributes: Dict[str, Any] = Field(default_factory=dict, example={"battery_life": "30 hours", "weight": "250g"})


class MatchRequest(BaseModel):
    item_a: ProductItem
    item_b: ProductItem


class MatchResponse(BaseModel):
    decision: MatchDecision
    similarity_score: float
    varying_features: list[str]
    normalized_item_a: Dict[str, Any]
    normalized_item_b: Dict[str, Any]
    processing_time_ms: float


class NormalizeRequest(BaseModel):
    title: str = Field(..., example="Hydro Flask 32oz Wide Mouth Bottle - Black")
    raw_attributes: Dict[str, Any] = Field(default_factory=dict, example={"capacity": "0.95 L", "color": "black"})


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "catalog-dedup"}


@app.post("/api/v1/match", response_model=MatchResponse)
def match_products(request: MatchRequest):
    t_start = time.perf_counter()
    try:
        decision, similarity, metadata = matcher.classify_pair(
            item_a=request.item_a.model_dump(),
            item_b=request.item_b.model_dump()
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return MatchResponse(
            decision=decision,
            similarity_score=metadata["similarity_score"],
            varying_features=metadata["varying_features"],
            normalized_item_a=metadata["normalized_a"],
            normalized_item_b=metadata["normalized_b"],
            processing_time_ms=round(elapsed_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/normalize", response_model=NormalizedAttributes)
def normalize_product_attributes(request: NormalizeRequest):
    try:
        return normalizer.normalize(
            raw_attributes=request.raw_attributes,
            title=request.title
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
