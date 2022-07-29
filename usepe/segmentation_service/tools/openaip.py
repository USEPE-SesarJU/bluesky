import httpx

_STORAGE_BUCKET = "https://storage.googleapis.com/29f98e10-a489-4c82-ae5e-489dbcd4912f"
_AIRSPACES = "asp"
_COUNTRY_CODE = "de"
_FORMAT = "txt"  # "geojson"


def download():
    with httpx.Client() as client:
        url = f"{_STORAGE_BUCKET}/{_COUNTRY_CODE}_{_AIRSPACES}.{_FORMAT}"
        pass
