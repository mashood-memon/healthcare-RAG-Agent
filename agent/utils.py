import requests

def geocode_location(location_text: str, geo_cache: dict) -> dict | None:
    cache_key = location_text.lower().strip()
    if cache_key in geo_cache:
        return geo_cache[cache_key]

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": location_text,
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
                "countrycodes": "us"
            },
            headers={"User-Agent": "HealthcareAgent/1.0"},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print(f"  [geocode] Warning: No results found for '{location_text}'")
            return None
            
        result = data[0]
        lat = float(result["lat"])
        lon = float(result["lon"])
        resolved_name = result.get("display_name", location_text)
        
        # Extract state code (e.g., "US-NC" -> "NC")
        state_code = None
        address = result.get("address", {})
        iso_code = address.get("ISO3166-2-lvl4", "")
        if iso_code and iso_code.startswith("US-"):
            state_code = iso_code.split("-")[1].upper()
            
        geo_cache[cache_key] = {"lat": lat, "lon": lon, "state_code": state_code}
        print(f"  [geocode] '{location_text}' → {resolved_name} ({lat:.4f}, {lon:.4f}) [State: {state_code}]")
        return geo_cache[cache_key]
        
    except Exception as e:
        print(f"  [geocode] Error: failed to geocode '{location_text}': {e}")
        return None
