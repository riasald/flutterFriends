"""Metadata schema definitions for butterfly data ingestion."""

RAW_REQUIRED_COLUMNS = [
    "source",
    "record_id",
    "genus",
    "scientific_name",
    "latitude",
    "longitude",
    "image_url",
]

RAW_OPTIONAL_COLUMNS = [
    "media_id",
    "institution_code",
    "catalog_number",
    "family",
    "species",
    "raw_species",
    "country",
    "state_province",
    "locality",
    "event_date",
    "basis_of_record",
    "license",
    "rights_holder",
    "image_format",
    "publisher",
    "dataset_key",
    "record_url",
    "local_image_path",
    "download_status",
    "download_error",
]

CLEAN_REQUIRED_COLUMNS = [
    "image_path",
    "latitude",
    "longitude",
    "species",
    "species_id",
]

CLEAN_OPTIONAL_COLUMNS = [
    "source",
    "record_id",
    "media_id",
    "specimen_key",
    "record_image_count",
    "family",
    "genus",
    "scientific_name",
    "country",
    "state_province",
    "locality",
    "event_date",
    "month",
    "basis_of_record",
    "license",
    "rights_holder",
    "image_url",
    "image_format",
    "record_url",
    "download_status",
    "download_error",
    "mask_path",
]
