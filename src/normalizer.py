import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class NormalizedAttributes(BaseModel):
    brand: Optional[str] = None
    color: Optional[str] = None
    storage_gb: Optional[int] = Field(None, description="Standardized storage capacity in GB")
    volume_ml: Optional[float] = Field(None, description="Standardized volume in milliliters")
    weight_grams: Optional[float] = Field(None, description="Standardized weight in grams")
    screen_size_inches: Optional[float] = Field(None, description="Standardized screen diagonal in inches")


class AttributeNormalizer:
    """
    Parses messy, unstandardized seller attribute keys and values
    into a canonical Pydantic schema with standardized metric units.
    """

    COLOR_PALETTE = {
        "black", "white", "silver", "gray", "grey", "gold", "blue", 
        "red", "green", "navy", "natural titanium", "space gray"
    }

    def normalize(self, raw_attributes: Dict[str, Any], title: str = "") -> NormalizedAttributes:
        combined_text = f"{title} " + " ".join(f"{k} {v}" for k, v in raw_attributes.items())
        combined_lower = combined_text.lower()

        return NormalizedAttributes(
            color=self._extract_color(combined_lower),
            storage_gb=self._extract_storage(combined_lower),
            volume_ml=self._extract_volume(combined_lower),
            weight_grams=self._extract_weight(combined_lower),
            screen_size_inches=self._extract_screen_size(combined_lower),
        )

    def _extract_color(self, text: str) -> Optional[str]:
        for color in self.COLOR_PALETTE:
            # Word boundary matching
            if re.search(r'\b' + re.escape(color) + r'\b', text):
                return color.title()
        return None

    def _extract_storage(self, text: str) -> Optional[int]:
        # Match patterns like: 128gb, 256 gb, 1tb
        match_tb = re.search(r'(\d+)\s*(?:tb|terabyte)', text)
        if match_tb:
            return int(match_tb.group(1)) * 1024

        match_gb = re.search(r'(\d+)\s*(?:gb|gigabyte)', text)
        if match_gb:
            return int(match_gb.group(1))
        return None

    def _extract_volume(self, text: str) -> Optional[float]:
        # Match liters: 0.95l, 1.5 l, 2 liter
        match_l = re.search(r'([\d.]+)\s*(?:l|liter|litres?)\b', text)
        if match_l:
            return round(float(match_l.group(1)) * 1000.0, 1)

        # Match milliliters: 500ml, 750 ml
        match_ml = re.search(r'([\d.]+)\s*(?:ml|milliliter)', text)
        if match_ml:
            return round(float(match_ml.group(1)), 1)

        # Match fluid ounces: 32 oz, 16.9 fl oz -> 1 fl oz ~ 29.5735 ml
        match_oz = re.search(r'([\d.]+)\s*(?:fl\s*oz|oz|ounce)', text)
        if match_oz:
            return round(float(match_oz.group(1)) * 29.5735, 1)

        return None

    def _extract_weight(self, text: str) -> Optional[float]:
        # Match kg: 0.25 kg
        match_kg = re.search(r'([\d.]+)\s*(?:kg|kilogram)', text)
        if match_kg:
            return round(float(match_kg.group(1)) * 1000.0, 1)

        # Match grams: 250 grams, 250g
        match_g = re.search(r'([\d.]+)\s*(?:g|gram|grams?)\b', text)
        if match_g:
            return round(float(match_g.group(1)), 1)

        return None

    def _extract_screen_size(self, text: str) -> Optional[float]:
        # Match 6.1 in, 6.1 inches, 6.1"
        match = re.search(r'([\d.]+)\s*(?:in|inch|inches|[\"\u201d])', text)
        if match:
            val = float(match.group(1))
            if 3.0 <= val <= 85.0:  # Sensible display size sanity check
                return val
        return None
